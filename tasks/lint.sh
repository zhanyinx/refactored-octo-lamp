#!/bin/bash

FAILURE=false

cd ../

# Security
safety check -r requirements.txt || FAILURE=true
bandit -ll -r spot_detection/ || FAILURE=true
bandit -ll -r training/ || FAILURE=true

# Typing
mypy spot_detection/  || FAILURE=true
mypy training/ || FAILURE=true

# Linting
pylint spot_detection/  || FAILURE=true
pylint training/ || FAILURE=true

# Codestyle
pycodestyle spot_detection/ || FAILURE=true
pycodestyle training/ || FAILURE=true

# Docstyle
pydocstyle spot_detection/ || FAILURE=true
pydocstyle training/ || FAILURE=true

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
