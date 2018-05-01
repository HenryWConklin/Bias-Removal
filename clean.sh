#!/bin/bash
python clean.py $1 > $1.clean
python remove_punctuation.py $1.clean | sort | uniq > ${1%.tsv}.nopunc.tsv
cut -f1 ${1%.tsv}.nopunc.tsv > ${1%.tsv}.before.txt
cut -f2 ${1%.tsv}.nopunc.tsv > ${1%.tsv}.after.txt
