import openai
from utils import generate_image, edit_image, process_dalle_images, affordance_func,extract_actions
import os
import requests
# Notebook Imports
from IPython.display import Image
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from pathlib import Path
from segment import GibsonSAM, ModelConfig
import argparse
from config import CLASSES


<<<<<<< HEAD
os.environ['OPENAI_API_KEY'] = ""
=======
os.environ['OPENAI_API_KEY'] = ""
>>>>>>> 91d4a15 (adding new changes)
openai.api_key = os.getenv("OPENAI_API_KEY")

HOME = "/home/aregbs/Desktop/gibson-afford"
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_hq_vit_h.pth")
SAM_ENCODER_VERSION  = "vit_h"




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_dir', help = 'file dir for saving edited generated data', required=True)
    parser.add_argument('--output_dir', help = 'file dir for saving edited generated data', required=True)
    parser.add_argument('--original_image', help = 'file path original image data', required=True)
    parser.add_argument('--variation_per_image', help = 'number of variation of base image', type=int, default=2)


    args = parser.parse_args()
    output_dir = args.output_dir
    original_image = args.original_image
    mask_dir = args.mask_dir
    variation_per_image = args.variation_per_image

   
    mask_dir= Path.cwd() / mask_dir

    mask_dir.mkdir(exist_ok=True)
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    
    config = ModelConfig(HOME=HOME, GROUNDING_DINO_CONFIG_PATH=GROUNDING_DINO_CONFIG_PATH,
                         GROUNDING_DINO_CHECKPOINT_PATH=GROUNDING_DINO_CHECKPOINT_PATH,
                         SAM_CHECKPOINT_PATH=SAM_CHECKPOINT_PATH,
                         SAM_ENCODER_VERSION=SAM_ENCODER_VERSION
                          )
    gibson = GibsonSAM(original_image, CLASSES, model_config= config)
    labels = gibson.get_image_annotations(mask_dir)
    labels = tuple(labels)

    print(f'labels: {labels}')

    paths = Path(mask_dir)
    paths = list(paths.iterdir())
    

    environment  = 'tabletop'

    query_template =  f"""Imagine {labels} on a {environment} 
    What are the possible interactions between these objects? Specify answer by giving names of Action, object performed on, instrument used to perform the action and the effect on the object. 
    An example is:
    ###
    Action:Peel
    Performed on: Onion 
    Instrument: knife
    Effect: peeled(onion)
    ### 
    present the result in the format of the example starting with ### at the begining Action
    """
        
    print(f'query_template: {query_template}')


    afford = affordance_func(template=query_template)
    print(f'affordance: {afford}')
    actions = extract_actions(afford, paths)
    print(f'actions: {actions}')
   
  
  
    for edit in actions:
        prompt = edit["Effect"]
        mask_path = edit["Performed on"]
        print(f"Performed on: {mask_path}")
        print(f"Effect: {prompt}\n")

        edit_response = edit_image(image_path=original_image, mask_dir=mask_path,variation_image=variation_per_image,prompt=prompt)
        edit_filepaths = process_dalle_images(edit_response, f"{prompt}", output_dir)
    


        
    
if __name__ == "__main__":
    main()
  
