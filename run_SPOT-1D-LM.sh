#!/bin/bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 FILE_LIST ESM_DEVICE PROTTRANS_DEVICE MODEL_DEVICE" >&2
    exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"

python "$here/generate_esm.py" --file_list "$1" --device "$2"
python "$here/generate_prottrans.py" --file_list "$1" --device "$3"
python "$here/run_inference.py" --file_list "$1" --device "$4"
