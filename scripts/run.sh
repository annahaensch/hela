#!/bin/bash

ROOT_DIR="$(git rev-parse --show-toplevel)"

COMMAND=${@:-bash}
docker run -it --rm      \
  -v /tmp:/tmp           \
  -v "${ROOT_DIR}":/hela \
  hela                   \
  ${COMMAND}
