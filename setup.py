import setuptools
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_string(string, rel_path="stepcount/__init__.py"):
    for line in read(rel_path).splitlines():
        if line.startswith(string):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError(f"Unable to find {string}.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stepcount",
    python_requires=">=3.8",
    version=get_string("__version__"),
    description="Step counter for wrist-worn accelerometers compatible with the UK Biobank Accelerometer Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OxWearables/stepcount",
    author=get_string("__author__"),
    author_email=get_string("__email__"),
    license=get_string("__license__"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
    ],
    packages=setuptools.find_packages(exclude=("test", "tests")),
    include_package_data=False,
    install_requires=[
        "actipy==2.0.1",
        "numpy>=1.22",
        "scipy>=1.9",
        "pandas>=1.4",
        "tqdm>=4.64",
        "joblib>=1.2.0",
        "scikit-learn==1.1.1",
        "imbalanced-learn==0.9.1",
        "hmmlearn==0.2.7",
        "torch~=1.12.1",
        "torchvision~=0.13.1",
        "transforms3d~=0.4.1"
    ],
    extras_require={
        "dev": [
            "flake8",
            "autopep8",
            "ipython",
            "ipdb",
            "twine",
        ],
        "docs": [
            "sphinx>=4.2",
            "sphinx_rtd_theme>=1.0",
            "readthedocs-sphinx-search>=0.1",
            "sphinxcontrib-programoutput>=0.17",
            "docutils<0.18",
        ],
    },
    entry_points={
        "console_scripts": [
            "stepcount=stepcount.stepcount:main"
        ]
    }
)
