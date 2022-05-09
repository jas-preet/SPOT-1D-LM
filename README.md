# SPOT-1D-LM
SPOT-1D-LM: Reaching Alignment-profile-based Accuracy in Predicting Protein Secondary and Tertiary Structural Properties without Alignment.

System Requirments
----

**Hardware Requirments:**
SPOT-1D-LM predictor has been tested on standard ubuntu 18 computer with approximately 32 GB RAM to support the in-memory operations.

* [Python3.7](https://docs.python-guide.org/starting/install3/linux/)
* [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive) (Optional if using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional if using GPU)

Installation
----

To install SPOT-1D-LM and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/jas-preet/SPOT-1D-LM.git`
2. `cd SPOT-1D-LM`

To download the model check points from the dropbox use the following commands in the terminal:

3. `wget https://servers.sparks-lab.org/downloads/SPOT-LM-checkpoints.xz`
4. `tar -xvf SPOT-LM-checkpoints.xz`

To install the dependencies and create a conda environment use the following commands

5. `conda create -n spot_1d_lm python=3.7`
6. `conda activate spot_1d_lm`

if GPU computer:
7. `conda install pytorch==1.7.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch`

for CPU only 
7. `conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch`

8. `pip install fair-esm`

9. `conda install pandas=1.1.1`

10. `conda install tqdm`

11. `pip install bio-embeddings[all]
`


Execute
----
To run SPOT-1D-LM use the following command

`bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cpu cpu cpu` to run model, ESM-1b and ProtTrans on cpu

or 

`bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cpu cpu cuda:0` to run model on gpu and, ESM-1b and ProtTrans on cpu

or

`bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cuda:0 cuda:1 cuda:2` to run model, ESM-1b and ProtTrans on gpu

Citation Guide
----
for more details on this work refer the manuscript

Please also cite and refer to ESM-1b and ProtTrans as the input used in this work is from these works. 
