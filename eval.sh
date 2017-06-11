#!/usr/bin/env bash

python gcn/train.py --model=gcn
python gcn/train.py --model=dense
python gcn/train.py --model=gcn_cheby