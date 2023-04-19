## Anaconda on Windows

1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (light-weight version of Anaconda). Choose **Miniconda3 Windows 64-bit**.
1. Install. Use the default recommended settings.
1. From the Start menu, search and open the **Anaconda Prompt**.

## Working with virtual environments

When you first open the Anaconda Prompt, you should see `(base)` in front of the prompt. This means that you are currently in the default "base" environment.

A best practice is to keep the base environment as minimal as possible and create a new separate environment every time you start a new project:
```console
$ conda create -n my-new-project python=3.9 git pip
```
In the above, we created a new environment called `my-new-project` with Python 3.9, Git, and Pip (the standard package installer for Python).

To activate the environment:
```console
$ conda activate my-new-project
```
After this, the prompt should change from `(base)` to `(my-new-project)`.

To exit the current environment:
```console
$ conda deactivate
```

To delete the environment:
```console
$ conda env remove -n my-new-project
```

To list existing environments:
```console
$ conda env list
```

## Why Anaconda?

I find Anaconda to be the easiest way to get started with Python on Windows. In many Linux systems, Python comes pre-installed and many purists prefer the standard tools to manage virtual environments like `venv` and `virtualenv`.

Still, there are cases where Anaconda is very convenient on Linux. For example,
the system may have an old version of Python but we don't have access to `sudo` or admin privileges to update it (e.g. we are using a company laptop, or logged into a compute cluster). Anaconda lets us install most data science tools without admin permissions.
This is because Anaconda is not merely an environment manager, but a platform on which we can do more things. For example, we can install R and Java JDK on it.

## Avoid mixing `conda` with `pip` to manage packages
Both Anaconda and Pip can be used to install Python packages: In Anaconda: `conda install my-package`, In Pip: `pip install my-package`.
However, it is recommend that you stick to one method (preferably Pip). My main workflow is as follows:

```bash
# Create a new virtual environment with Python, Git, and Pip
$ conda create -n awesome-project python=3.9 git pip
$ conda activate awesome-project
# Install packages using Pip
$ pip install awesome-package
$ ...
```

That is, use Anaconda only for the environment management (create, activate, deactivate, delete, etc.) and setup of high-level tools (e.g. Python, Git, Pip, Java, etc.).
