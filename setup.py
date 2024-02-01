import sys
import os.path
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.insert(0, os.path.dirname(__file__))

import setuptools
import codecs

import versioneer


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_string(string, rel_path="src/stepcount/__init__.py"):
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
    python_requires=">=3.8, <3.11",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Step counter for wrist-worn accelerometers compatible with the UK Biobank Accelerometer Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OxWearables/stepcount",
    download_url="https://github.com/OxWearables/stepcount",
    author=get_string("__author__"),
    maintainer=get_string("__maintainer__"),
    maintainer_email=get_string("__maintainer_email__"),
    license=get_string("__license__"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    packages=setuptools.find_packages(where="src", exclude=("test", "tests")),
    package_dir={"": "src"},
    include_package_data=False,
    install_requires=[
        "actipy>=3.0.5",
        "numpy==1.24.*",
        "scipy==1.10.*",
        "pandas==2.0.*",
        "tqdm==4.64.*",
        "joblib==1.2.*",
        "scikit-learn==1.1.1",
        "imbalanced-learn==0.9.1",
        "hmmlearn==0.3.*",
        "torch==1.13.*",
        "torchvision==0.14.*",
        "transforms3d==0.4.*"
    ],
    extras_require={
        "dev": [
            "flake8",
            "autopep8",
            "ipython",
            "ipdb",
            "twine",
            "tomli",
            "jupyter",
            "matplotlib",
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
            "stepcount=stepcount.stepcount:main",
            "stepcount-collate-outputs=stepcount.utils.collate_outputs:main",
            "stepcount-generate-commands=stepcount.utils.generate_commands:main"
        ]
    }
)
