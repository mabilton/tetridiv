#!/bin/sh

# Runs Docker container with 'python3 -m pytest .' instead of Jupyter Lab:
docker run -v "$(pwd)":/root --entrypoint python3 mabilton/tetridiv -m pytest .