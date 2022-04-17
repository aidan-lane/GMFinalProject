#!/bin/sh

export PYTHONPATH='.:gen'
python3 -m client.main "$@"
unset PYTHONPATH