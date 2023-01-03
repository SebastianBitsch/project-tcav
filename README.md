# Project-TCAV
## About
A framework built on top of the original work from Been Kim in "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)" See https://github.com/tensorflow/tcav. 

The framework includes - among other things - funtionality for running the original work using bash on linux, generating and running tcav on noisy versions of the original images, as well as classifing the same images using GoogleNet to compare accuracy with the tcav interpretability score and saliency maps.

The target concept of zebra, as well as the concepts of striped, zigzagged and dotted are already downloaded and available in the tcav/data/ directory in clean images and for 10 different noise levels. Additional data can be downloaded and/or generated using `download_and_make_datasets.py` and `saltandpepper.ipynb`

## Usage
The main functionality can be run like so

**Recreating original paper results:** To recreate the original paper results the file `RECREATE_RESULTS.ipynb` has been used.

**Generate images with salt/pepper noise:** Generate noisy versions of the target images using `saltandpepper.ipynb`.

**Run TCAV Framework for a given folder of data:** This can be done using the `RUNTCAV.py` and passed arguments based on the needs and data available in tcav/data. Generated TCAV scores for the concepts and saves results.

**Run TCAV Framework for many folders at once:** Useful when getting results for images of many different noise levels. Can be run in `RUN_ALL_TCAVS.py`.

**Plot results:** Many different results can be plotted in `plot_resuts.py`.

**Compute saliency maps:** Saliency maps for images can be computed and visualized in `saliency_maps.ipynb`. <span style="color:red">Please note that the code for computing saliency maps is mostly copied from the github repository at: https://github.com/sunnynevarekar/pytorch-saliency-maps</span>
