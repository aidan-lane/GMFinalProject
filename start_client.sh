#!/bin/sh

export PYTHONPATH='.:./gen'
python3 client/main.py "$@"
unset PYTHONPATH