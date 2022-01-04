### To run it on HPC run: bsub < hpc_training.sh
### General options
# !/bin/sh
### –- specify queue --
#BSUB -q gpuv100
#BSUB -gpu "num=1"
### -- set the job Name --
#BSUB -J TrainPilot
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 3:00

### -- request 32GB of system-memory
#BSUB -R "rusage[mem=32GB]"

### - notification email
#BSUB -u greg.papaspiropoulos@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

echo "Running script..."

python3 train.py