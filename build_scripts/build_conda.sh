#!/bin/bash

# Be sure to have packaged for PyPI first
conda skeleton pypi actipy --output-dir conda-recipe
conda build conda-recipe/actipy
# anaconda login
# anaconda upload --user oxwear /path/to/package.tar.bz2
