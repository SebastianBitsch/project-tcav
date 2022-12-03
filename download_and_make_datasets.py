# run on mac
# https://developer.apple.com/metal/tensorflow-plugin/

# python3 download_and_make_datasets.py --num_images=100 --num_folders=5

import subprocess
import os
import argparse
from tensorflow.io import gfile
import imagenet_and_broden_fetcher as fetcher

def make_concepts_targets_and_randoms(source_dir: str, number_of_images_per_folder: int, number_of_random_folders: int, imagenet_classes: list, broden_concepts: list):
    # Run script to download data to source_dir
    if not gfile.exists(os.path.join(source_dir, 'broden1_224/')) or not gfile.exists(os.path.join(source_dir, 'inception5h')):
        print("Running FetchDataAndModels.sh to get broden1_224, inception5h and mobilenet")
        subprocess.call(['bash', 'FetchDataAndModels.sh', source_dir])

    # make targets from imagenet
    imagenet_dataframe = fetcher.make_imagenet_dataframe("./imagenet_url_map.csv")
    for image in imagenet_classes:
        print(f"Downloading {number_of_images_per_folder} images to data/{image}")
        fetcher.fetch_imagenet_class(source_dir, image, number_of_images_per_folder, imagenet_dataframe)

    # Make concepts from broden
    for concept in broden_concepts:
        print(f"Downloading images for concept: {concept}")
        fetcher.download_texture_to_working_folder(
            broden_path=os.path.join(source_dir, 'broden1_224'),
            saving_path=source_dir,
            texture_name=concept,
            number_of_images=number_of_images_per_folder
        )

    print("Starting function call fetcher.generate_random_folders")
    # Make random folders. If we want to run N random experiments with tcav, we need N+1 folders.
    fetcher.generate_random_folders(
        working_directory=source_dir,
        random_folder_prefix="random500",
        number_of_random_folders=number_of_random_folders+1,
        number_of_examples_per_folder=number_of_images_per_folder,
        imagenet_dataframe=imagenet_dataframe
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create examples and concepts folders.')
    
    parser.add_argument('--num_images', type=int, help='Number of images to be included in each folder')
    parser.add_argument('--num_folders', type=int, help='Number of folders with random examples that we will generate for tcav')
    parser.add_argument("--targets", type=str, nargs="+", default=["zebra"], help="The name of imagenet classes to use, defaults to zebra")
    parser.add_argument("--concepts", type=str, nargs="+", default=['striped', 'dotted', 'zigzagged'], help="The broden concepts to use, defaults to 'striped, dotted, zigzagged'")

    args = parser.parse_args()
    
    source_dir = "tcav/data"

    # create folder if it doesn't exist
    if not gfile.exists(source_dir):
        gfile.makedirs(os.path.join(source_dir))
        print("Created source directory at " + source_dir)
    
    # Make data
    make_concepts_targets_and_randoms(source_dir, args.num_images, args.num_folders, args.targets, args.concepts)
    print("Successfully created data at " + source_dir)
