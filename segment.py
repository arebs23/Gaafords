import os
import supervision as sv
from dataclasses import dataclass
import torch
import torch.nn as nn
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import cv2
from typing import List
import numpy as np
from tqdm import tqdm
import random
import math
from PIL import Image
from pathlib import Path



@dataclass
class ModelConfig:
    HOME: str 
    GROUNDING_DINO_CONFIG_PATH: str 
    GROUNDING_DINO_CHECKPOINT_PATH: str 
    SAM_CHECKPOINT_PATH: str 
    SAM_ENCODER_VERSION: str



class GibsonSAM:
    BOX_TRESHOLD = 0.40
    TEXT_TRESHOLD = 0.25
    MIN_IMAGE_AREA_PERCENTAGE = 0.002
    MAX_IMAGE_AREA_PERCENTAGE = 0.80
    APPROXIMATION_PERCENTAGE = 0.75
    def __init__(self, image_path: str, class_names: List[str], model_config, device: str = "cuda"):
        self.device = device
        self.image_path = image_path
        self.class_names = class_names
        # self.visualize_results = visualize_results
        self.model_config = model_config
        self.model_config = model_config
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()
        self.grounding_dino_model = Model(model_config_path=self.model_config.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.model_config.GROUNDING_DINO_CHECKPOINT_PATH)
        self.sam = sam_model_registry[self.model_config.SAM_ENCODER_VERSION](checkpoint=self.model_config.SAM_CHECKPOINT_PATH).to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)



    @staticmethod
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    

    @staticmethod
    def enhance_class_name(class_names: List[str]) -> List[str]:
        return [
            f"{class_name}"
            for class_name
            in class_names
    ]
    
    def get_image_annotations(self, mask_dir):
        image = cv2.imread(self.image_path)
      
        detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=self.enhance_class_name(self.class_names),
                box_threshold=self.BOX_TRESHOLD,
                text_threshold=self.TEXT_TRESHOLD
            )
        detections = detections[detections.class_id != None]
        
        detections.mask = self.segment(
            self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy )
        titles = [self.class_names[class_id]for class_id in detections.class_id]
        
      
        for i, image_array in enumerate(detections.mask):
           
            chosen_mask = image_array.astype("uint8")
            chosen_mask[chosen_mask != 0] = 255
            chosen_mask[chosen_mask == 0] = 1
            chosen_mask[chosen_mask == 255] = 0
            chosen_mask[chosen_mask == 1] = 255
            width = 512
            height = 512
            mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))  # create an opaque image mask

            # Resize the chosen_mask to match the desired dimensions
            chosen_mask_resized = cv2.resize(chosen_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Convert mask back to pixels to add our mask replacing the third dimension
            pix = np.array(mask)
            pix[:, :, 3] = chosen_mask_resized

        # Convert pixels back to an RGBA image and display
            new_mask = Image.fromarray(pix, "RGBA")
            new_mask.save(os.path.join(mask_dir, f"{titles[i]}.png"))
        
        return titles

# if __name__ == "__main__":
#     CLASSES = ["mug", "rubber plate", "apple", "tomatoes", "knife", "bowl"]
#     # CLASSES = ['car', 'dog', 'person', 'chair', 'shoe', 'belt', 'cup']
#     image_path= "/home/aregbs/Desktop/gibson-afford/gen_data/output/1.png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-05-29T08%3A49%3A48Z&ske=2023-05-30T08%3A49%3A48Z&sks=b&skv=2021-08-06&sig=s43fP0Jtqh7L3FuSRysguIVsOURTOguyDEUcU1%2Blnz4%3D.png"
#     output_folder = "new_mask"
#     DATA_DIR = Path.cwd() / output_folder

#     DATA_DIR.mkdir(exist_ok=True)
   
#     gibson = GBSONDATA(image_path, CLASSES)
#     gibson.get_image_annotations(DATA_DIR)
  