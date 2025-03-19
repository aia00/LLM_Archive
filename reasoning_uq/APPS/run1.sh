#!/usr/bin/env bash

for i in $(seq 0 14)
do
  echo "Running question index $i"
  python uncertainty_test.py --index $i
done
