#!/bin/bash

python generate_esm.py --file_list $1 --device $2
python generate_prottrans.py --file_list $1 --device $3
python run_inference.py --file_list $1 --device $4
