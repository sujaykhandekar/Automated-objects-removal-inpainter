# Automated-objects-removal-inpainter

Automated object remover Inpainter is a project that combines Semantic segmentation and EdgeConnect arhitectures with minor changes in order to remove specified object/s from photos. For Semantic Segmentation, the code from pytorch has been adapted, whereas for EdgeConnect, the code has been adapted from [https://github.com/knazeri/edge-connect](https://github.com/knazeri/edge-connect).

This project is capable of removing objects from list of 20 different ones.It can be use as photo editing tool as well as for Data augmentation.

Python 3.7.7 and pytorch 1.5.1 have been used in this project.

## How does it work?

<img src="https://user-images.githubusercontent.com/31131069/89242660-188c8e80-d5d0-11ea-8277-1ed6b9a0f83d.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/31131069/89242666-1cb8ac00-d5d0-11ea-83b0-61c86d26fa68.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/31131069/89242673-1fb39c80-d5d0-11ea-8ac4-906b5d06d4d6.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/31131069/89242684-25a97d80-d5d0-11ea-9756-811189856ae4.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/31131069/89242689-28a46e00-d5d0-11ea-9041-70bd16103d17.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/31131069/89242695-2b9f5e80-d5d0-11ea-8c72-c865cc72616b.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/31131069/89242813-6b664600-d5d0-11ea-9125-276610c10bda.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/31131069/89242821-6f926380-d5d0-11ea-997b-005e5875471a.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/31131069/89242827-728d5400-d5d0-11ea-9776-e3392312e774.png" width="30%"></img> 

Semantic segmenator model of deeplabv3/fcn resnet 101 has been combined with EdgeConnect. A pre-trained segmentation network has been used for object segmentation (generating a mask around detected object), and its output is fed to a EdgeConnect network along with input image with portion of mask removed. EdgeConnect uses two stage adversarial architecture where first stage is edge generator followed by image completion network. EdgeConnect paper can be found [here](https://arxiv.org/abs/1901.00212) and code in this [repo](https://github.com/knazeri/edge-connect)




## Prerequisite
* python 3
* pytorch 1.0.1 <
* NVIDIA GPU + CUDA cuDNN (optional)

## Installation
* clone this repo 
```
git clone https://github.com/sujaykhandekar/Automated-objects-removal-inpainter.git
cd Automated-objects-removal-inpainter
```
or alternately downlaod zip file.
* install latest pytorch version from [https://pytorch.org/](https://pytorch.org/)
* install other python requirements using this command
```
pip install -r requirements.txt
```
* Install one of the three pretrained Edgeconnect model and copy them in ./checkpoints directory  
[Plcaes2](https://drive.google.com/drive/folders/1KyXz4W4SAvfsGh3NJ7XgdOv5t46o-8aa) (option 1)
[CelebA](https://drive.google.com/drive/folders/1nkLOhzWL-w2euo0U6amhz7HVzqNC5rqb) (option 2)
[Paris-street-view](https://drive.google.com/drive/folders/1cGwDaZqDcqYU7kDuEbMXa9TP3uDJRBR1) (option 3)

or alternately you can use this command:
```
bash ./scripts/download_model.sh
```

## prediction/Test
For quick prediction you can run this command. If you don't have cuda/gpu please run the second command.
```
python test.py --input ./examples/my_small_data --output ./checkpoints/resultsfinal --remove 3 15
```
It will take sample images in the ./examples/my_small_data  directory and will create and produce result in directory ./checkpoints/resultsfinal. You can replace these input /output directories with your desired ones.
numbers after --remove specifies objects to be removed in the images. ABove command will remove 3(bird) and 15(people) from the images. Check segmentation-classes.txt for all removal options along with it's number.

Output images will all be 256x256. It takes around 10 mintues for 1000 images on NVIDIA GeForce GTX 1650

for better quality but slower runtime you can use  this command
```
python test.py --input ./examples/my_small_data --output ./checkpoints/resultsfinal --remove 3 15 --cpu yes
```
It will run the segmentation model on cpu. It will be 5 times slower than on gpu (default)
For other options inclduing different segmentation model and EdgeConnect parameters to change please make corresponding modifications in .checkpoints/config.yml file

## training
For trainig your own segmentation model you can refer to this [repo](https://github.com/CSAILVision/semantic-segmentation-pytorch) and replace .src/segmentor_fcn.py with your model.

For training Edgeconnect model plaese refer to orignal [EdgeConnect repo](https://github.com/knazeri/edge-connect)  after training you can copy your model weights in .checkpoints/ 

## some results
<img src="https://user-images.githubusercontent.com/31131069/89245607-2db8eb80-d5d7-11ea-94e1-e16ac6be8009.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245666-4f19d780-d5d7-11ea-8a0e-12ffc9367cba.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245571-1974ee80-d5d7-11ea-91ce-e6c95ea8d8fc.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245452-d450bc80-d5d6-11ea-968a-b0fd60c4d3ad.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245970-18908c80-d5d8-11ea-9e99-b91245052870.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245711-6a84e280-d5d7-11ea-8eea-fd718b500799.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89246127-6a391700-d5d8-11ea-85a3-20d65ab3a571.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245762-8b4d3800-d5d7-11ea-89f6-16c21142b2bd.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245794-a5871600-d5d7-11ea-8426-d3bddeed3dd5.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245812-b2a40500-d5d7-11ea-80e4-6a65c9fd3ae7.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245841-c3547b00-d5d7-11ea-8fa2-aecd9dceef0a.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245870-ce0f1000-d5d7-11ea-87a2-0ded6c355fe5.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245885-dd8e5900-d5d7-11ea-8aec-c1a35b7a604e.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89245898-e5e69400-d5d7-11ea-9147-5467ba36f14b.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89251524-40d2b800-d5e5-11ea-9a6e-cadf96d8ef5b.png" width="23%"></img> <img src="https://user-images.githubusercontent.com/31131069/89251550-521bc480-d5e5-11ea-8906-d0bdad16d641.png" width="23%"></img> 

## Next steps
*  pretrained EdgeConnect models used in this project are trained on 256 x256 images. To make output images of the same size as input two approaches can be used. You can train your own Edgeconnect model on bigger images.Or you can create subimages of 256x256 for every object detected in the image and then merge them back together after passing through edgeconnect to reconstruct orignal sized image.Similar approach has been used in this [repo](https://github.com/javirk/Person_remover)
* To detect object not present in segmentation classes , you can train your own segmentation model or you can use pretrained segmentation models from this [repo](https://github.com/CSAILVision/semantic-segmentation-pytorch), which has 150 different categories available.
* It is also possible to combine opnecv's feature matching and edge prediction from EdgeConnect to highlight and create mask for relvant objects based on single mask created by user. I may try this part myself.

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International.](https://creativecommons.org/licenses/by-nc/4.0/)

Except where otherwise noted, this content is published under a [CC BY-NC](https://github.com/knazeri/edge-connect) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## Citation
```
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}

@InProceedings{Nazeri_2019_ICCV,
  title = {EdgeConnect: Structure Guided Image Inpainting using Edge Prediction},
  author = {Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
  month = {Oct},
  year = {2019}
}
```
