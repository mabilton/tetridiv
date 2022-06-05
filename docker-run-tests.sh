#!/bin/sh

docker run -d --name tetridiv_test -v "$(pwd)":/root mabilton/tetridiv || exit

until [ "`docker inspect -f {{.State.Running}} tetridiv_test`"=="true" ]; do
    sleep 0.1;
done;

docker exec tetridiv_test pip install --force-reinstall . && \
docker exec tetridiv_test python3 -m pytest . && \
docker stop tetridiv_test

until [ "`docker inspect -f {{.State.Running}} tetridiv_test`"=="false" ]; do
    sleep 0.1;
done;

docker rm tetridiv_test
