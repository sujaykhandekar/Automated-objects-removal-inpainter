import argparse
import os
import random
from shutil import copyfile

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms
from PIL import Image


# cog
from cog import BasePredictor, Input, Path

from src.config import Config
from src.edge_connect import EdgeConnect

import tempfile 


# Maps object to index
obj2idx = {
    "Background":0, "Aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, "bus":6, "car":7, "cat":8,
    "chair":9, "cow":10, "dining table":11, "dog":12, "horse":13, "motorbike":14, "person":15,
    "potted plant":16, "sheep":17, "sofa":18, "train":19, "tv/monitor":20
    }



def load_config(mode=None, objects_to_remove=None):
    print('Object(s) to remove:', objects_to_remove)
    

    # load config file
    path = "./checkpoints"
    config_path = os.path.join(path, "config.yml")

    # create checkpoints path if does't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile("./config.yml.example", config_path)

    # load config file
    config = Config(config_path)

    # test mode
    config.MODE = mode
    config.MODEL = 3
    config.OBJECTS = objects_to_remove
    config.SEG_DEVICE = "cuda"
    config.INPUT_SIZE = 256

    # outputs
    if not os.path.exists("./results_images"):
        os.makedirs("./results_images")
    config.RESULTS = "./results_images"
    return config


# Instantiate Cog Predictor
class Predictor(BasePredictor):
    def setup(self):

        # Select torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, 
        image_path: Path = Input(description="Input image (ideally a square image)"),
        objects_to_remove: str = Input(description="Object(s) to remove (separate with comma, e.g. car,cat,bird). See full list of names at https://github.com/sujaykhandekar/Automated-objects-removal-inpainter/blob/master/segmentation_classes.txt", default='person,car'),

        ) -> Path:

        # format input image
        image_path = str(image_path)
        image = Image.open(image_path).convert('RGB') 
        image.save(image_path) # resave formatted image

        # parse objects to remove
        objects_to_remove = objects_to_remove.split(',') 
        objects_to_remove = [obj2idx[x] for x in objects_to_remove]

        mode = 2  # 1: train, 2: test, 3: eal
        self.config = load_config(mode, objects_to_remove=objects_to_remove)
        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)

        # initialize random seed
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)

        # save to path 
        self.config.TEST_FLIST = image_path

        # build the model and initialize
        model = EdgeConnect(self.config)
        model.load()

        # model test
        output_image = model.test()
        output_image = output_image.cpu().numpy()
        output_image = Image.fromarray(np.uint8(output_image)).convert('RGB')
        
        # save output image as Cog Path object
        output_path = Path(tempfile.mkdtemp()) / "output.png"
        output_image.save(output_path)
        print(output_path)
        return output_path
