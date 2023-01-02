import subprocess
import time

"""
File for running the tcav framework on different datasets in one call.
Here we use it to generate results for all of our noisy data (10% to 100%) using the RUNTCAV.py file
The results are saved to to json in results/
"""

# Run the noise data from 10% to 100%

for i in range(10):
    
    start = time.time()

    data_dir = f"data_noise{i+1}0"

    print(f"**** STARTING TO RUN {data_dir} ****")

    subprocess.call(['python3', 'RUNTCAV.py', "--data_dir", data_dir])
    end = time.time()

    print(f"**** FINISHED RUNNING {data_dir}: IN {str(end - start)} ****")
