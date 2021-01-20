#!/bin/bash
ROOT_DIR="$(readlink -f "$(dirname "$(readlink -f "$0")")/..")"

COMMAND=${@:-bash}
docker run -it --rm      \
  -v /tmp:/tmp           \
  -v "${ROOT_DIR}":/hela \
  hela                   \
  ${COMMAND}
