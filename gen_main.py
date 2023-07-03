import openai
from utils import generate_image
import os
import requests
from utils import get_completion, generate_image
# Notebook Imports
from IPython.display import Image
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from pathlib import Path
from gen_template import prompt, template
import argparse


os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', help = 'file dir for saving generated data',  required=True)
    parser.add_argument('--original_image', help = 'number of original image', type = int, default=1)
    parser.add_argument('--variation_per_image', help = 'number of variation of base image', type=int, default=3)
    args = parser.parse_args()
   

    gen_dir = args.gen_dir
    original_image = args.original_image
    variation_per_image = args.variation_per_image
    prompts = get_completion(template=template, prompt=prompt)
    print(prompts)
    generate_image(base_images_number=original_image, labels=prompts, output_folder=gen_dir, variation_per_image=variation_per_image)

if __name__ == "__main__":
    main()
  