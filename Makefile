# Makefile for creating a new release of the package and uploading it to PyPI

PYTHON = python3

help:
	@echo "Use 'make upload' to upload the package to PyPi"


upload:
	rm -r dist build | true
	$(PYTHON) setup.py sdist bdist_wheel
	twine upload dist/*

# For testing:
test-upload:
	rm -r dist build | true
	$(PYTHON) setup.py sdist bdist_wheel
	twine upload --repository testpypi dist/*
