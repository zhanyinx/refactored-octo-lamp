#!/bin/bash

FAILURE=false

cd ../


# Security
safety check -r requirements.txt || FAILURE=true
bandit -ll -r spot_detection/ || FAILURE=true
bandit -ll -r training/ || FAILURE=true
bandit -ll -r evaluation/ || FAILURE=true
bandit -ll -r api/ || FAILURE=true

# Typing
mypy spot_detection/  || FAILURE=true
mypy training/ || FAILURE=true
mypy evaluation/ || FAILURE=true
mypy api/ || FAILURE=true

# Linting
pylint spot_detection/  || FAILURE=true
pylint training/ || FAILURE=true
pylint evaluation/ || FAILURE=true
pylint api/ || FAILURE=true

# Codestyle
pycodestyle spot_detection/ || FAILURE=true
pycodestyle training/ || FAILURE=true
pycodestyle evaluation/ || FAILURE=true
pycodestyle api/ || FAILURE=true

# Docstyle
pydocstyle spot_detection/ || FAILURE=true
pydocstyle training/ || FAILURE=true
pydocstyle evaluation/ || FAILURE=true
pydocstyle api/ || FAILURE=true

# Testing
# cd ./spot_detection/tests/
# pytest || FAILURE=true
# cd ../../

#cd ./training/tests/
#pytest || FAILURE=true
#cd ../../

cd ./tasks

if [ "$FAILURE" = true ]; then
  echo "Linting failed."
else
  echo "Linting succeeded."
fi
