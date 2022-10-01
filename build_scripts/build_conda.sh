#!/bin/bash

# Be sure to have packaged for PyPI first

conda install anaconda-client &&
conda install conda-build &&
# TODO: allow user-specified version and append this to next line: --version x.x.x
conda skeleton pypi stepcount --output-dir conda-recipe &&
conda build -c conda-forge conda-recipe/stepcount

printf "\nNext steps:\n-----------\n"
printf "Login to Anaconda:\n> anaconda login\n"
printf "\nUpload package (path is printed in previous steps):\n> anaconda upload --user oxwear /path/to/package.tar.bz2\n\n"

# anaconda login
# anaconda upload --user oxwear /path/to/package.tar.bz2
