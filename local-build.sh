#!/bin/bash

set -e
PACKAGE_NAME=rs_course
cd doc
make clean html coverage
cat build/coverage/python.txt
cd ..
pycodestyle --max-doc-length 160 --ignore E402,E203,E501,W503 ${PACKAGE_NAME}
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
mypy --config-file mypy.ini ${PACKAGE_NAME}
export TEST_ON_GPU=
pytest --cov ${PACKAGE_NAME} --cov-report term-missing \
       --cov-fail-under=95 ${PACKAGE_NAME}
scc -i py ${PACKAGE_NAME}
