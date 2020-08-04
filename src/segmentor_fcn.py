from torchvision import models
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import numpy as np
from imageio import imread
from skimage.color import rgb2gray, gray2rgb
import cv2

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

def decode_segmap(image,objects,nc=21):
                
    r = np.zeros_like(image).astype(np.uint8)
    for l in objects:
        idx = image == l
        r[idx] = 255#fill  r with 255 wherever class is 1 and so on
    return np.array(r)


def fill_gaps(values):
    searchval=[255,0,255]
    searchval2=[255,0,0,255]
    idx=(np.array(np.where((values[:-2]==searchval[0]) & (values[1:-1]==searchval[1]) & (values[2:]==searchval[2])))+1)
    idx2=(np.array(np.where((values[:-3]==searchval2[0]) & (values[1:-2]==searchval2[1]) & (values[2:-1]==searchval2[2]) & (values[3:]==searchval2[3])))+1)
    idx3=(idx2+1)
    new=idx.tolist()+idx2.tolist()+idx3.tolist()
    newlist = [item for items in new for item in items]
    values[newlist]=255
    return values

def fill_gaps2(values):
    searchval=[0,255]
    searchval2=[255,0]
    idx=(np.array(np.where((values[:-1]==searchval[0]) & (values[1:]==searchval[1]))))
    idx2=(np.array(np.where((values[:-1]==searchval[0]) & (values[1:]==searchval[1])))+1)
    
    new=idx.tolist()+idx2.tolist()
    newlist = [item for items in new for item in items]
    values[newlist]=255
    return values


def remove_patch_og(real_img,mask):
    og_data = real_img.copy()
    idx = mask == 255  ### cutting out mask part from real image here
    og_data[idx] =255
    return og_data




def segmentor(seg_net,img,dev,objects):
    #plt.imshow(img); plt.show()
    if seg_net==1:
        net=fcn
    else:
        net=dlab
    if dev == 'cuda':
        trf = T.Compose([T.Resize(400),
                 #T.CenterCrop(224),
        T.ToTensor(), 
        T.Normalize(mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225])])
    else:
        trf = T.Compose([T.Resize(680),
                 #T.CenterCrop(224),
        T.ToTensor(), 
        T.Normalize(mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    mask=decode_segmap(om,objects)
    height,width =mask.shape
    img=np.array(img.resize((width, height), Image.ANTIALIAS))


    og_img=remove_patch_og(img,mask)
    #plt.imshow(mask); plt.show()
    return og_img,mask


    
    
    
       
    
    
    