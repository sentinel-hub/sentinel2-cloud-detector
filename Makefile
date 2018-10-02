# Makefile for creating a new release of the package and uploading it to PyPI

PYTHON = python3

help:
	@echo "Use 'make upload' to upload the package to PyPi"


upload:
	$(PYTHON) setup.py sdist
	twine upload dist/*

# For testing:
test-upload:
	$(PYTHON) setup.py sdist
	twine upload --repository testpypi dist/*
