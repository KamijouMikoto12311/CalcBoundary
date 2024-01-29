#!/bin/bash
rm -rf alltogetherperimeter.dat
for f in [sr]*/; do
    tail -n 1000 $f/perimeter.dat >> altogetherperimeter.dat
done