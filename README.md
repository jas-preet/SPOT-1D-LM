# SPOT-1D-LM

*Reaching Alignment-profile-based Accuracy in Predicting Protein Secondary and Tertiary Structural Properties without Alignment.*

SPOT-1D-LM predicts per-residue structural properties from a protein sequence. It uses embeddings from the ESM-1b and ProtTrans language models instead of a multiple sequence alignment.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Limitations](#limitations)
- [Dataset](#dataset)
- [Tests](#tests)
- [Citation](#citation)

## Requirements

- Python 3 (tested with 3.12) and the packages in `requirements.txt`
- About 16 GB of RAM to generate embeddings on CPU (measured peak: 12.7 GB the first time ProtT5 loads, then 9.9 GB with sequences up to 1,022 residues)
- About 1 GB of RAM for the prediction step on its own (measured peak: 0.7 GB for a 118-residue protein)
- About 30 GB of disk space, because ESM-1b (7 GB) and ProtT5 (21 GB) are downloaded the first time they run
- Optional: an NVIDIA GPU with CUDA

## Installation

```bash
git clone https://github.com/jas-preet/SPOT-1D-LM.git
cd SPOT-1D-LM
pip install -r requirements.txt
```

Download the model checkpoints (about 1 GB) and check the download. The server only offers plain HTTP, so don't skip the checksum.

```bash
curl -LO http://183.36.5.251:8080/sparks_downloads/servers/downloads/Resource/Protein/2_Protein_local_structure_prediction/spot_1d_lm.tar.gz
echo "594e5a140a335cca89640a6bd3c56e8a30e4ebee7e226433b7ce1461721aa97e  spot_1d_lm.tar.gz" | shasum -a 256 -c
tar -xzf spot_1d_lm.tar.gz spot_1d_lm/checkpoints/SPOT-LM-checkpoints.xz
tar -xf spot_1d_lm/checkpoints/SPOT-LM-checkpoints.xz
```

On Linux, use `sha256sum -c` instead of `shasum -a 256 -c`. If the link stops working, look for SPOT-1D-LM on the [Zhou lab download page](https://zhouyq-lab.szbl.ac.cn/download/).

## Usage

The pipeline takes a file list, then the device for ESM-1b, ProtTrans and the SPOT-1D-LM models, in that order:

```bash
bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cpu cpu cpu
bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cpu cpu cuda:0        # only SPOT-1D-LM on GPU
bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cuda:0 cuda:1 cuda:2  # everything on GPU
```

Embeddings are saved in `inputs/` and results in `results/`. The file list has one FASTA path per line, and each FASTA file must contain one sequence. Lower-case letters are accepted, a trailing `*` is removed, and residues other than the 20 standard amino acids are treated as unknown.

To try the prediction step without the language models, use the embeddings included for `5LOS_A`:

```bash
python run_inference.py --file_list file_lists/test_file_list.txt --device cpu
```

`run_inference.py` also accepts `--save_path` for the output directory and `--checkpoints_dir` for checkpoints stored somewhere other than `checkpoints/`.

## Output

Each input produces `results/<fasta file name>.csv`, with one row per residue. See `results/5LOS_A.csv` for an example.

| Column | Meaning |
| --- | --- |
| `AA` | Amino acid |
| `SS3`, `SS8` | Predicted 3-state and 8-state secondary structure |
| `ASA` | Accessible surface area |
| `HseU`, `HseD` | Half-sphere exposure, up and down |
| `CN` | Contact number |
| `Psi`, `Phi`, `Theta`, `Tau` | Backbone angles in degrees |
| `P3C`, `P3E`, `P3H` | 3-state probabilities |
| `P8C`, `P8S`, `P8T`, `P8H`, `P8G`, `P8I`, `P8E`, `P8B` | 8-state probabilities |

Earlier versions wrote raw model scores to the `P3` and `P8` columns. They now hold probabilities that sum to 1. The predicted states are unchanged.

Unknown residues have no reference surface area, so their `ASA` value (between 0 and 1) is a placeholder and does not mean the residue is buried.

## Limitations

- ESM-1b accepts up to 1,022 residues, so longer sequences are rejected.
- Proteins are processed one at a time, because this implementation has only been tested with a batch size of 1.
- The training code is not available.

## Dataset

The download above also contains the data behind this work: `fasta`, `dssp`, `hse` and `theta` folders with 39,484 files each, plus train, validation and test lists in `lists`. To extract everything, including the checkpoints archive:

```bash
tar -xzf spot_1d_lm.tar.gz
```

## Tests

The tests use random weights and stand-ins for the language models, so they don't need any downloads.

```bash
pip install pytest
pytest
```

## Citation

Singh J, Paliwal K, Litfin T, Singh J, Zhou Y. Reaching alignment-profile-based accuracy in predicting protein secondary and tertiary structural properties without alignment. *Scientific Reports*. 2022;12:7607. [doi:10.1038/s41598-022-11684-w](https://doi.org/10.1038/s41598-022-11684-w)

Please also cite [ESM-1b](https://github.com/facebookresearch/esm) and [ProtTrans](https://github.com/agemagician/ProtTrans), which provide the input embeddings.
