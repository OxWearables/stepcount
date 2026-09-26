"""Helpers for the self-supervised learning model."""

from __future__ import annotations

from os import PathLike
from typing import Any, Callable, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from transforms3d.axangles import axangle2mat
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import Adam  # type: ignore[attr-defined]  # Export is absent from PyTorch stubs.

from stepcount._types import NDArray


Device = Union[str, torch.device]
DatasetItem = Tuple[torch.Tensor, Any, Any]

verbose = True
torch_cache_path = Path(__file__).parent / 'torch_hub_cache'

torch.multiprocessing.set_sharing_strategy('file_system')  # type: ignore[no-untyped-call]  # Missing from PyTorch stubs.


class RandomSwitchAxis:
    """
    Randomly permute the three axes in a raw sample.
    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        x = sample[0, :]
        y = sample[1, :]
        z = sample[2, :]

        choice = random.randint(1, 6)

        if choice == 1:
            sample = torch.stack([x, y, z], dim=0)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=0)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=0)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=0)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=0)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=0)

        return sample


class RotationAxis:
    """
    Rotate a sample around a random axis.
    """

    def __call__(self, sample: NDArray) -> NDArray:
        sample = np.swapaxes(sample, 0, 1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 0, 1)
        return sample


class RandomDecimation:
    """
    Randomly decimate and restore the input along the time axis.
    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        decimation_factor = random.randint(1, 3)
        T = sample.shape[1]
        sample = sample[:, ::decimation_factor]

        sample = torch.nn.functional.interpolate(
            sample.unsqueeze(0),
            size=T,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        return sample


class NormalDataset(Dataset[DatasetItem]):
    def __init__(
        self,
        X: NDArray,
        y: Optional[NDArray] = None,
        pid: Optional[NDArray] = None,
        name: str = "",
        augmentation: bool = False,
        transpose_channels_first: bool = True,
    ) -> None:

        X = X.astype(
            "f4"
        )  # PyTorch defaults to float32

        if transpose_channels_first:
            X = np.transpose(X, (0, 2, 1))
        self.X = torch.from_numpy(X)

        self.y: Optional[torch.Tensor]
        if y is not None:
            self.y = torch.tensor(y)
        else:
            self.y = None

        self.pid: Optional[NDArray] = pid

        if augmentation:
            self.transform = transforms.Compose([RandomSwitchAxis(), RotationAxis(), 
                                                 RandomDecimation()])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> DatasetItem:
        index: Any = idx.tolist() if isinstance(idx, torch.Tensor) else idx

        sample = self.X[index, :]

        y_value: Any
        if self.y is not None:
            y_value = self.y[index]
        else:
            y_value = np.NaN

        pid_value: Any
        if self.pid is not None:
            pid_value = self.pid[index]
        else:
            pid_value = np.NaN

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, y_value, pid_value


class InferenceDataset(Dataset[DatasetItem]):
    """Read selected windows lazily and cast only the current sample."""

    def __init__(
        self,
        X: NDArray,
        indices: Optional[NDArray] = None,
    ) -> None:
        self.X = X
        self.indices = None if indices is None else np.asarray(indices)
        can_share = X.dtype == np.float32 and all(stride >= 0 for stride in X.strides)
        self.tensor = torch.from_numpy(X) if can_share else None

    def __len__(self) -> int:
        return len(self.X) if self.indices is None else len(self.indices)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> DatasetItem:
        index: Any = idx.tolist() if isinstance(idx, torch.Tensor) else idx
        source_index = index if self.indices is None else self.indices[index]
        if self.tensor is not None:
            sample = self.tensor[source_index].transpose(0, 1)
            return sample, np.NaN, np.NaN
        array = self.X[source_index].astype("f4", copy=False)
        if any(stride < 0 for stride in array.strides):
            array = np.ascontiguousarray(array)
        array = np.transpose(array, (1, 0))
        return torch.from_numpy(array), np.NaN, np.NaN


class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
        path: Union[str, PathLike[str]] = "checkpoint.pt",
        trace_func: Callable[[str], None] = print,
    ) -> None:
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

        self.path = path

    def __call__(self, val_loss: float, model: nn.Module) -> None:

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f" ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_sslnet(
    tag: str = 'v1.0.0',
    pretrained: bool = False,
    repo_path: Optional[Union[str, PathLike[str]]] = None,
) -> nn.Module:
    """
    Load and return the Self Supervised Learning (SSL) model from pytorch hub.

    :param str tag: Tag on the ssl-wearables repo to check out
    :param bool pretrained: Initialise the model with UKB self-supervised pretrained weights.
    :param repo_path: Path to a local copy of the ssl-wearables repo. When provided, uses this
        directly with source='local', skipping GitHub download and cache lookup.
        Note: when repo_path is provided, the tag parameter is ignored — the caller
        is responsible for ensuring the local repo matches the expected version.
    :return: pytorch SSL model
    :rtype: nn.Module
    """

    if repo_path is not None:
        sslnet = cast(
            nn.Module,
            torch.hub.load(  # type: ignore[no-untyped-call]  # torch.hub is untyped upstream.
                str(repo_path), 'harnet10', trust_repo=True, source='local',
                class_num=2, pretrained=pretrained, verbose=verbose,
            ),
        )
        return sslnet

    repo_name = 'ssl-wearables'
    repo = f'OxWearables/{repo_name}:{tag}'

    if not torch_cache_path.exists():
        Path.mkdir(torch_cache_path, parents=True, exist_ok=True)

    torch.hub.set_dir(str(torch_cache_path))  # type: ignore[no-untyped-call]  # torch.hub is untyped upstream.

    # Reuse a matching local checkout to avoid an unnecessary network fetch.
    cache_dirs = [f for f in torch_cache_path.iterdir() if f.is_dir()]
    repo_path = next((f for f in cache_dirs if repo_name in f.name and tag in f.name), None)

    if repo_path is None:
        repo_path = repo
        source = 'github'
    else:
        repo_path = str(repo_path)
        source = 'local'
        if verbose:
            print(f'Using local {repo_path}')

    sslnet = cast(
        nn.Module,
        torch.hub.load(  # type: ignore[no-untyped-call]  # torch.hub is untyped upstream.
            repo_path, 'harnet10', trust_repo=True, source=source, class_num=2,
            pretrained=pretrained, verbose=verbose,
        ),
    )
    return sslnet


def predict(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: Device,
    output_logits: bool = False,
    collect_metadata: bool = True,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Iterate over the dataloader and do prediction with a pytorch model.

    :param nn.Module model: pytorch Module
    :param DataLoader data_loader: pytorch dataloader
    :param str device: pytorch map device
    :param bool output_logits: When True, output the raw outputs (logits) from the last layer (before classification).
                                When False, argmax the logits and output a classification scalar.
    :param bool collect_metadata: When False, skip retaining labels and participant IDs during inference.
    :return: true labels, model predictions, pids
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """


    predictions_list: list[torch.Tensor] = []
    true_list: list[torch.Tensor] = []
    pid_list: list[Any] = []
    model.eval()

    if len(dataloader) == 0:
        return np.array([]), np.array([]), np.array([])

    with torch.inference_mode():
        for x, y, pid in tqdm(dataloader, total=len(dataloader), mininterval=5, disable=not verbose, bar_format='Classifying segments: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]'):
            x = x.to(device, dtype=torch.float)
            logits = cast(torch.Tensor, model(x))
            if collect_metadata:
                true_list.append(y)
                pid_list.extend(pid)
            if output_logits:
                predictions_list.append(logits.cpu())
            else:
                pred_y = torch.argmax(logits, dim=1)
                predictions_list.append(pred_y.cpu())

    predictions_tensor = torch.cat(predictions_list)
    true_values = torch.flatten(torch.cat(true_list)).numpy() if collect_metadata else np.array([])
    pid_values = np.array(pid_list) if collect_metadata else np.array([])
    prediction_values = predictions_tensor.numpy() if output_logits else torch.flatten(predictions_tensor).numpy()
    return true_values, prediction_values, pid_values


def train(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    device: Device,
    class_weights: Optional[Sequence[float]] = None,
    weights_path: Union[str, PathLike[str]] = 'weights.pt',
    num_epoch: int = 100,
    learning_rate: float = 0.0001,
    patience: int = 5,
) -> nn.Module:
    """
    Iterate over the training dataloader and train a pytorch model.
    After each epoch, validate model and early stop when validation loss function bottoms out.

    Trained model weights will be saved to disk (weights_path).

    :param nn.Module model: pytorch model
    :param DataLoader train_loader: training data loader
    :param DataLoader val_loader: validation data loader
    :param str device: pytorch map device
    :param class_weights: Array of training class weights to use with weighted cross entropy loss.
                        Leave empty to use unweighted loss.
    :param weights_path: save location for the trained weights (state_dict)
    :param num_epoch: number of training epochs
    :param learning_rate: Adam learning rate
    :param patience: early stopping patience
    """
    optimizer = Adam(
        model.parameters(), lr=learning_rate, amsgrad=True
    )

    if class_weights is not None:
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(
        patience=patience, path=weights_path, verbose=verbose, trace_func=print
    )

    for epoch in range(num_epoch):
        model.train()
        train_losses: list[torch.Tensor] = []
        train_acces: list[torch.Tensor] = []
        for x, y, _ in tqdm(train_loader, total=len(train_loader), disable=not verbose, bar_format='Training: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]'):
            x.requires_grad_(True)
            x = x.to(device, dtype=torch.float)
            true_y = y.to(device, dtype=torch.long)

            optimizer.zero_grad()

            logits = cast(torch.Tensor, model(x))
            loss = cast(torch.Tensor, loss_fn(logits, true_y))
            loss.backward()  # type: ignore[no-untyped-call]  # Missing from PyTorch stubs.
            optimizer.step()

            pred_y = torch.argmax(logits, dim=1)
            train_acc = torch.sum(pred_y == true_y)
            train_acc = train_acc / (pred_y.size()[0])

            train_losses.append(loss.cpu().detach())
            train_acces.append(train_acc.cpu().detach())

        val_loss, val_acc = _validate_model(model, val_loader, device, loss_fn)

        epoch_len = len(str(num_epoch))
        print_msg = (
            f"[{epoch:>{epoch_len}}/{num_epoch:>{epoch_len}}] | "
            + f"train_loss: {np.mean(train_losses):.3f} | "
            + f"train_acc: {np.mean(train_acces):.3f} | "
            + f"val_loss: {val_loss:.3f} | "
            + f"val_acc: {val_acc:.2f}"
        )

        early_stopping(val_loss, model)

        if verbose:
            print(print_msg)

        if early_stopping.early_stop:
            if verbose:
                print('Early stopping')
                print(f'SSLNet weights saved to {weights_path}')
            break

    return model


def _validate_model(
    model: nn.Module,
    val_loader: DataLoader[Any],
    device: Device,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """Return mean loss and accuracy over a validation data loader."""
    model.eval()
    losses: list[torch.Tensor] = []
    acces: list[torch.Tensor] = []
    with torch.inference_mode():
        for x, y, _ in val_loader:
            x = x.to(device, dtype=torch.float)
            true_y = y.to(device, dtype=torch.long)

            logits = cast(torch.Tensor, model(x))
            loss = cast(torch.Tensor, loss_fn(logits, true_y))

            pred_y = torch.argmax(logits, dim=1)

            val_acc = torch.sum(pred_y == true_y)
            val_acc = val_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach())
            acces.append(val_acc.cpu().detach())
    loss_array = np.array(losses)
    accuracy_array = np.array(acces)
    return float(np.mean(loss_array)), float(np.mean(accuracy_array))
