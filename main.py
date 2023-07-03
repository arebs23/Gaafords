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
    HOME: str = "/home/aregbs/Desktop/gibson-afford"
    GROUNDING_DINO_CONFIG_PATH: str = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH: str = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
    SAM_CHECKPOINT_PATH: str = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    IMAGES_DIRECTORY: str = "/home/aregbs/Desktop/gibson-afford/Objects Lab Instance Segmentation.v2i.coco-segmentation/test"
    ANNOTATIONS_DIRECTORY = os.path.join(HOME, 'new_annotate')
    SAM_ENCODER_VERSION: str = "vit_h"
    SAVE_PATH: str = os.path.join(HOME, "visualize_path")




class GBSONDATA:
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    MIN_IMAGE_AREA_PERCENTAGE = 0.002
    MAX_IMAGE_AREA_PERCENTAGE = 0.80
    APPROXIMATION_PERCENTAGE = 0.75
    def __init__(self, image_dir: str, class_names: List[str], device: str = "cuda"):
        self.device = device
        self.image_dir = image_dir
        self.class_names = class_names
        # self.visualize_results = visualize_results
        self.model_config = ModelConfig()
        self._images = {}
        self._annotations = {}
        self._plot_images = []
        self._plot_titles = []
        self.model_config = ModelConfig()
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
            f"all {class_name}s"
            for class_name
            in class_names
        ]
    
    
    def get_image_annotations(self, mask_dir, image_extensions = ['jpg', 'jpeg', 'png']):
        image_paths = sv.list_files_with_extensions(
            directory=self.model_config.IMAGES_DIRECTORY, 
            extensions=image_extensions)

        for image_path in tqdm(image_paths):
            image_name = image_path.name
            image_path = str(image_path)
            image = cv2.imread(image_path)

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
                xyxy=detections.xyxy
            )
            new_mask = Image.fromarray(detections.mask, "RGBA")
            new_mask.save(mask_dir, 'classes.png')
            self._images[image_name] = image
            self._annotations[image_name] = detections

            
    
   
    
    def plot_image_with_mask(self, value: int):
        for image_name, detections in random.sample(self._annotations.items(), value):
            image = self._images[image_name]
            self._plot_images.append(image)
            self._plot_titles.append(image_name)

            labels = [
                f"{self.class_names[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _ 
                in detections]
            annotated_image = self.mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            self._plot_images.append(annotated_image)
            title = " ".join(set([
                self.class_names[class_id]
                for class_id
                in detections.class_id
            ]))
            self._plot_titles.append(title)
          

        sv.plot_images_grid(
            save_path=self.model_config.SAVE_PATH,
            images=self._plot_images,
            titles=self._plot_titles,
            grid_size=(len(self._annotations), 2),
            size=(2 * 4, len(self._annotations) * 4)
        )
        
    
    def save_annotations(self):
        sv.Dataset(
            classes=self.class_names,
            images=self._images,
            annotations=self._annotations
        ).as_pascal_voc(
            annotations_directory_path=self.model_config.ANNOTATIONS_DIRECTORY,
            min_image_area_percentage=self.MIN_IMAGE_AREA_PERCENTAGE,
            max_image_area_percentage=self.MAX_IMAGE_AREA_PERCENTAGE,
            approximation_percentage=self.APPROXIMATION_PERCENTAGE
        )


if __name__ == "__main__":
    CLASSES = ["mug", "rubber plate", "plate", "porceline", "apple", "tomatoes", "knife", "bowl"]
    # CLASSES = ['car', 'dog', 'person', 'chair', 'shoe', 'belt', 'cup']
    image_directory= "/home/aregbs/Desktop/gibson-afford/DATA_GENERATIVE_MODEL"
    output_folder = "mask"
    DATA_DIR = Path.cwd() / output_folder

    DATA_DIR.mkdir(exist_ok=True)
   
    gibson = GBSONDATA(ModelConfig.IMAGES_DIRECTORY, CLASSES)
    gibson.get_image_annotations(DATA_DIR)
    gibson.plot_image_with_mask(4)