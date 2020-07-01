#!/bin/bash

FAILURE=false

cd ../


# Security
safety check -r requirements.txt || FAILURE=true
bandit -ll -r spot_detection/ || FAILURE=true
bandit -ll -r evaluation/ || FAILURE=true

# Typing
mypy spot_detection/  || FAILURE=true
mypy evaluation/ || FAILURE=true

# Linting
pylint spot_detection/  || FAILURE=true
pylint evaluation/ || FAILURE=true

# Codestyle
pycodestyle spot_detection/ || FAILURE=true
pycodestyle evaluation/ || FAILURE=true

# Docstyle
pydocstyle spot_detection/ || FAILURE=true
pydocstyle evaluation/ || FAILURE=true

# Testing
# cd ./spot_detection/tests/
# pytest || FAILURE=true
# cd ../../

#cd ./training/tests/
#pytest || FAILURE=true
#cd ../../

cd ./bin

if [ "$FAILURE" = true ]; then
  echo "Linting failed."
else
  echo "Linting succeeded."
fi
