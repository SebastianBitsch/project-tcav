import os
import json
import argparse

import tcav.activation_generator as act_gen

import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils

# python3 tcav_hpc.py --num_exp=5 --save_name=results_100_5_20221129

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument("--num_exp", type=int, help="The number of random experiments, the same as the number of random folders")
    parser.add_argument("--save_name", type=str, help="The name to save the results to. DONT ADD .JSON!")
    
    parser.add_argument("--model_to_run", type=str, nargs="?", default="GoogleNet", help="The model to run, defaults to GoogleNet")
    parser.add_argument("--target", type=str, nargs="?", default="zebra", help="The taget of the model, defaults to zebra")
    parser.add_argument("--bottlenecks", type=str, nargs="+", default=['mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b'], help="The layers to look at")
    parser.add_argument("--concepts", type=str, nargs="+", default=['striped', 'dotted', 'zigzagged'], help="The broden concepts to use, defaults to 'striped, dotted, zigzagged'")
    
    args = parser.parse_args()
    print("Arguments: ")
    for arg in vars(args):
        print("--", arg,": ", getattr(args, arg))
    
    model_to_run = args.model_to_run
    
    # where activations are stored (only if your act_gen_wrapper does so)
    working_dir = os.chmod()
    activation_dir =  working_dir + '/tcav/activations/'
    cav_dir = working_dir + '/tcav/cavs/'
    source_dir = working_dir + '/tcav/data/'

    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(cav_dir)

    sess = utils.create_session()
    
    GRAPH_PATH = 'tcav/data/inception5h/tensorflow_inception_graph.pb'
    LABEL_PATH = 'tcav/data/inception5h/imagenet_comp_graph_label_strings.txt'

    mymodel = model.GoogleNetWrapper_public(sess, GRAPH_PATH, LABEL_PATH)
    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=120)
    
    mytcav = tcav.TCAV(
        sess = sess,
        target = args.target,
        concepts = args.concepts,
        bottlenecks = args.bottlenecks,
        activation_generator = act_generator,
        alphas = [0.1],
        cav_dir = cav_dir,
        num_random_exp = args.num_exp
        #random_concepts=? se original Run_TCAV.ipynb
    )
    
    print('This may take a while... Go get corny!')
    results = mytcav.run(run_parallel=False)
    print('done! - check /results/')

    with open(f"results/{args.save_name}.json", "w") as outfile:
        json.dump(results, outfile)
