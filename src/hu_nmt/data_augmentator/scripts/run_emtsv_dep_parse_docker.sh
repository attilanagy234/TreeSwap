#!/bin/bash

cat ../data/hun_input.txt | docker run -i mtaril/emtsv tok-dep > ../data/hun_output.txt