#!/bin/bash

set -e
PACKAGE_NAME=rs_course
cd doc
make clean html coverage
cat build/coverage/python.txt
cd ..
flake8 ${PACKAGE_NAME} scripts tests
pydocstyle ${PACKAGE_NAME} scripts tests
pylint ${PACKAGE_NAME} scripts tests
mypy ${PACKAGE_NAME} scripts tests
export TEST_ON_GPU=
pytest
scc -i py ${PACKAGE_NAME} scripts
