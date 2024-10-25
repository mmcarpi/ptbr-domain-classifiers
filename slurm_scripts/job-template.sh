#!/bin/bash -l

# Set the slurm output file, this is where all command line output is redirected to. %j is replaced by the job id
#SBATCH --output=slurm_out_%j.txt

# Define computational resources. This job requests 8 CPUs and 1 GPU in a single node.
#SBATCH -n 8 # Cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1

# Sepecify the partition (arandu for the DGX-A100 and devwork for the workstations)
#SBATCH -p arandu

# Print the name of the worker node to the output file
echo "Running on"
hostname

# Copy files from the home folder to the output folder
cp -R /home/[myuser]/[myproject] /output/[myuser]/

# Call Docker and run the code
docker run --user "$(id -u):$(id -g)" --rm -v /output/[myuser]/[myproject]:/workspace/data nvcr.io/nvidia/pytorch:22.11-py3 -w /workspace/data python3 code/main.py

# Move the results to the home folder (Temporarily disabled)
# mv /output/[myuser]/[myproject]/results/* /home/[myuser]/[myproject]/results/

# Clean the output folder
rm -r /output/[myuser]/[myproject]

echo "Done"
