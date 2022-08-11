#!/bin/bash

#
#SBATCH --job-name=test
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=10:00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=2048    # Memory in MB per cpu allocated

hostname
python main.py
exit
