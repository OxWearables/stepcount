from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed_status(label: str, verbose: bool) -> Iterator[None]:
    if not verbose:
        yield
        return

    print(f"{label}...", end="\r", flush=True)
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    print(f"{label}... Done! ({elapsed:.2f}s)")
