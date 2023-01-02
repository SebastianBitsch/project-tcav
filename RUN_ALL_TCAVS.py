import subprocess
import time

# Run the noise data from 10% to 100%
for i in range(10):
    
    start = time.time()

    data_dir = f"data_noise{i+1}0"

    print(f"**** STARTING TO RUN {data_dir} ****")

    subprocess.call(['python3', 'RUNTCAV_NOISY.py', "--data_dir", data_dir])
    end = time.time()

    print(f"**** FINISHED RUNNING {data_dir}: IN {str(end - start)} ****")
