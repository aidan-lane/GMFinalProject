#!/bin/sh

export PYTHONPATH='.:gen'
python3 -m server.main "$@"
unset PYTHONPATH