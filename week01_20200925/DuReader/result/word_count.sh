#!/bin/bash

if [ $# -lt 1 ];then
    echo "$0 <input_file>"
    exit 1
fi

python word_count.py --prepare --input_files=$1 |sort|uniq -c|awk '{print $2,$1}'|sort -k2n