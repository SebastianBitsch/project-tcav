"""
File used to generate TCAV scores for a single folder of data.
We use it for our 10 folders of zebras with different noise levels. It is however not called
directly but from the file RUN_ALL_TCAVS.py

We have fixed the number of images and the number of experiments at 120 and 5, and remain there
for all expereiments.

Takes a data-directory as input which is expected to hold the zebra images. See argparser help.
"""

import tcav.activation_generator as act_gen
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils
import json
import shutil
import argparse
import absl
import os

absl.logging.set_verbosity(0)

# Global parameters - these are set and not changed in any of our experiments for consistency
num_random_exp = 5
num_images = 120
bottlenecks = ['mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']
target = 'zebra'
concepts = ["dotted","striped","zigzagged"]

# this is a regularizer penalty parameter for linear classifier to get CAVs. 
alphas = [0.1]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument("--data_dir", type=str, help="the name of the folder the data is stored in (/tcav/tcav_examples/image_models/imagenet/data/XXXX). Results will be saved under same name")
    
    args = parser.parse_args()

    working_dir = os.getcwd()
    activation_dir =  working_dir + '/activations/'
    cav_dir = working_dir + '/cavs/'

    source_dir = working_dir + '/tcav/tcav_examples/image_models/imagenet/data/' + args.data_dir

    # Empty activations and cavs before every run
    shutil.rmtree(activation_dir)
    shutil.rmtree(cav_dir)
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(cav_dir)

    # Create TensorFlow session.
    sess = utils.create_session()

    GRAPH_PATH = "tensorflow_inception_graph.pb"
    LABEL_PATH = "imagenet_comp_graph_label_strings.txt"

    _model = model.GoogleNetWrapper_public(sess, GRAPH_PATH, LABEL_PATH)

    act_generator = act_gen.ImageActivationGenerator(_model, source_dir, activation_dir, max_examples=num_images)

    _tcav = tcav.TCAV(sess,
                    target,
                    concepts,
                    bottlenecks,
                    act_generator,
                    alphas,
                    cav_dir=cav_dir,
                    num_random_exp=num_random_exp)

    print ('This may take a while... Go get corny!')
    
    results = _tcav.run(run_parallel=False)
    print ('done!')

    # Save the resulting scores
    with open(f"results/results_{args.data_dir}_{num_images}_{num_random_exp}.json", "w") as outfile:
        json.dump(results, outfile)

