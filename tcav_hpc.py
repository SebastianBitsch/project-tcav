# SKAL LIGGE I ØVERSTE MAPPE
import json
import argparse

import tcav.activation_generator as act_gen

import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument("--num_exp", type=int, help="The number of random experiments, the same as the number of random folders")
    parser.add_argument("--save_filename", type=str, help="The name to save the results to. DONT ADD .JSON!")
    
    # parser.add_argument("--working_dir", type=str, nargs="?", default="tcav", help="The path to the /tcav folder. i.e. /zhome/c9/c/156514/Desktop/project-tcav-master/tcav")
    parser.add_argument("--data_path", type=str, nargs="?", default="data", help="The name of the folder where the data is stored, defaults to data")
    parser.add_argument("--model_to_run", type=str, nargs="?", default="GoogleNet", help="The model to run, defaults to GoogleNet")
    parser.add_argument("--target", type=str, nargs="?", default="zebra", help="The taget of the model, defaults to zebra")
    parser.add_argument("--bottlenecks", type=str, nargs="+", default=['mixed3a','mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b'], help="The layers to look at")
    parser.add_argument("--concepts", type=str, nargs="+", default=['striped', 'dotted', 'zigzagged'], help="The broden concepts to use, defaults to 'striped, dotted, zigzagged'")
    
    args = parser.parse_args()

    model_to_run = args.model_to_run
    working_dir = "tcav" #args.working_dir
    
    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir + '/activations/'
    cav_dir = working_dir + '/cavs/'
    source_dir = working_dir + f'/{args.data_path}/'

    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)

    sess = utils.create_session()

    
    GRAPH_PATH = f'{working_dir}/{args.data_path}/inception5h/tensorflow_inception_graph.pb'
    LABEL_PATH = f'{working_dir}/{args.data_path}/inception5h/imagenet_comp_graph_label_strings.txt'

    mymodel = model.GoogleNetWrapper_public(sess, GRAPH_PATH, LABEL_PATH)
    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)

    mytcav = tcav.TCAV(
        sess = sess,
        target = args.target,
        concepts = args.concepts,
        bottlenecks = args.bottlenecks,
        activation_generator = act_generator,
        alphas = [0.1],
        cav_dir = cav_dir,
        num_random_exp = args.num_exp
    )
    
    print('This may take a while... Go get corny!')
    results = mytcav.run(run_parallel=False)
    print('done!')


    with open(f"results/{args.save_filename}.json", "w") as outfile:
        json.dump(results, outfile)
