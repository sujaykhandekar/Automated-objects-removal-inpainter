import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect


def main(mode=None):
    r"""starts the model

    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = EdgeConnect(config)
    model.load()


    
    # model test
    print('\nstart testing...\n')
    model.test()

    

def load_config(mode=None):
    r"""loads model config

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    # test mode
    parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
    parser.add_argument('--output', type=str, help='path to the output directory')
    parser.add_argument('--remove', nargs= '*' ,type=int, help='objects to remove')
    parser.add_argument('--cpu', type=str, help='machine to run segmentation model on')
    args = parser.parse_args()
    
    #if path for checkpoint not given
    if args.path is None:
        args.path='./checkpoints'
    config_path = os.path.join(args.path, 'config.yml')
    
       # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

   
    # test mode
    config.MODE = 2
    config.MODEL = args.model if args.model is not None else 3
    config.OBJECTS = args.remove if args.remove is not None else [3,15]
    config.SEG_DEVICE = 'cpu' if args.cpu is not None else 'cuda'
    config.INPUT_SIZE = 256
    if args.input is not None:
        config.TEST_FLIST = args.input
    
    if args.edge is not None:
        config.TEST_EDGE_FLIST = args.edge
    if args.output is not None:
        config.RESULTS = args.output
    else: 
        if not os.path.exists('./results_images'):
            os.makedirs('./results_images')
        config.RESULTS = './results_images'
    
    
      
    
    
    return config


if __name__ == "__main__":
    main()
