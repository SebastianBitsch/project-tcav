import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils
import json
import tensorflow as tf
import shutil
import argparse

import os

num_random_exp = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument("--data_dir", type=str, help="the name of the folder the data is stored in. Results will be saved under same name")
    
    args = parser.parse_args()

    working_dir = os.getcwd() #"/Users/sebastianbitsch/Downloads/tcav-master/"

    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir + '/activations/'
    cav_dir = working_dir + '/cavs/'

    source_dir = working_dir + '/tcav/tcav_examples/image_models/imagenet/MEGADATA/' + args.data_dir
    bottlenecks = ['mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']  # @param 

    # Empty activations and cavs before every run
    shutil.rmtree(activation_dir)
    shutil.rmtree(cav_dir)
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(cav_dir)

    # this is a regularizer penalty parameter for linear classifier to get CAVs. 
    alphas = [0.1]

    target = 'zebra'
    concepts = ["dotted","striped","zigzagged"]

    # Create TensorFlow session.
    sess = utils.create_session()


    GRAPH_PATH = "tensorflow_inception_graph.pb"
    LABEL_PATH = "imagenet_comp_graph_label_strings.txt"

    mymodel = model.GoogleNetWrapper_public(sess, GRAPH_PATH, LABEL_PATH)

    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=120)

    import absl
    absl.logging.set_verbosity(0)

    mytcav = tcav.TCAV(sess,
                    target,
                    concepts,
                    bottlenecks,
                    act_generator,
                    alphas,
                    cav_dir=cav_dir,
                    num_random_exp=num_random_exp)
    print ('This may take a while... Go get corny!')
    results = mytcav.run(run_parallel=False)
    print ('done!')

    with open(f"results/results_{args.data_dir}_120_5.json", "w") as outfile:
        json.dump(results, outfile)

