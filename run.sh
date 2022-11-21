#!/bin/bash
#BSUB -J TCAV
#BSUB -o tcav_%J.out
#BSUB -e tcav_%J.err
#BSUB -q hpc
#BSUB -W 12:00
#BSUB -N 2
#BSUB -R “rusage[mem=2GB]”
#BSUB -u s204163@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -Ne
python3 download_and_make_datasets.py --number_of_images_per_folder=2 --number_of_random_folders=2
python3 tcav_hpc.py --num_exp=2 --save_file_name=results_20221121