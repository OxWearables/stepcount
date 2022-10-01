#!/bin/bash

python setup.py sdist bdist_wheel &&
twine check dist/* &&
printf "\nTo upload to Test PyPI:\n> twine upload --repository-url https://test.pypi.org/legacy/ dist/*\n" &&
printf "\nTo upload to PyPI:\n> twine upload dist/*\n\n"

# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
