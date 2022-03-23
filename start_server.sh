#!/bin/sh

export PYTHONPATH='.:./gen'
python3 server/main.py "$@"
unset PYTHONPATH