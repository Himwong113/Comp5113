from engine import evaluate
from main import get_args_parser
import torch
import numpy as np
from torch import nn
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from models import build_model
from torchvision.models import resnet50
import argparse
from hubconf import detr_resnet50,_make_detr,detr_resnet101
from util.box_ops import box_xyxy_to_cxcywh,box_cxcywh_to_xyxy
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from getcat import getactboxmapver3
import json
from imageget import getimage
torch.set_grad_enabled(False)

### plot the att

def rescaled_bbox(out_bbox,size):
    img_w, img_h =size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b*torch.tensor([img_w,img_h, img_w,img_h],dtype=torch.float32)
    return b

def detect(im, model, transforms):
    img =transforms(im).unsqueeze(0)
    ## make image to be a tensor
    output = model(img)
    ##generate the prediction
    #print(output['pred_logits'].shape)
    #print(output['pred_logits'].softmax(-1).shape)
    #print(output['pred_logits'].softmax(-1))
    #output['pred_logits'] Shape: (batch_size, num_queries, num_classes)#(1,100,91)
    probas = output['pred_logits'].softmax(-1)[0,:,:-1]
    #print(probas.shape)
    keep = probas.max(-1).values>0.85   #keep class >0.85 only
    #print(keep.shape)#
    bboxes_scaled = rescaled_bbox(output['pred_boxes'][0,keep],im.size)

    return probas[keep],bboxes_scaled,output['pred_logits'],keep,probas

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return e_x / e_x.sum()

with open('./datasets/annotations/instances_val2017.json', 'r') as file:
    text = json.load(file)
categories=text['categories']
coco_id = [item['id'] for item in categories]
coco_data = [item['name'] for item in categories]
coco_classes={f'{coco_id[i]}':coco_data[i] for i in range(len(coco_id))}

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
[0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transforms = T.Compose([
    T.Resize(800), # make the picture to resize row=800
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def SVMget(imageid):
    padded_string = str(imageid).zfill(12) #image id = padding(0)with (12-imagid len) + imageid
    image_path=f'.\datasets\\val2017\\{padded_string}.jpg' #39769torch.nonzero(class_detected).flatten()
    im=Image.open(image_path)
    ## call pretrain model from hugconf
    detr= detr_resnet101(pretrained=True)

    score,boxes,logits,keep,probas=detect(im,detr,transforms)


    ##### encoder attention
    conv_feature, enc_attn_weights,dec_attn_weights =[],[],[]

    hooks =[
        detr.backbone[-2].register_forward_hook(
            lambda self,input ,output:conv_feature.append(output)
        ),
        detr.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda  self,input,output : enc_attn_weights.append(output[1]) # grab the last layer in encoder from MLA #ie take the attention weighted
        ),
        detr.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda  self,input,output : dec_attn_weights.append(output[1]) #take the MHA layer which have qkv form encode and qkv from decoder # ie 2nd MHA in endcoder
        )
    ]
    img =transforms(im).unsqueeze(0)
    output =detr(img)

    for hook in hooks:
        hook.remove()
    conv_feature=conv_feature[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    #print(conv_feature[0].tensor.shape) #batch,Channel,h,w #1,2048,h,w
    #print(enc_attn_weights.shape)   # batch, matrix_h (query x key), matrix_w (query x key) # 1,850,850
    #print(dec_attn_weights) # batch, matrix_h (queryfrom dec x key from end), matrix_w (queryfrom dec x key from end)#1,100,850




    # get the feature map shape
    h, w = conv_feature['0'].tensors.shape[-2:]#h = 25 ,w=34
    #fig, axs = plt.subplots( ncols=1 , nrows=1,figsize=(22,7))
    #colors = COLORS * 100

    buffer = torch.zeros(h*w)
    for idx in keep.nonzero():
        ###get all the hig confidence query vector and sum up them
        ten = dec_attn_weights[0, idx]
        buffer += ten[0]






    for i, item in enumerate(buffer):
        if buffer[i] >= 0.005:
            buffer[i] = 1
        else:
            buffer[i] = 0


    # Reshape buffer
    # Plot each point in buffer_f
    # Show the raw image  and in shape (h,w)
    return h,w,buffer

#imageid=2006
#1503 31, 61, 71, 92




"""
imageid=2153

h,w,buffer= SVMget(imageid)
im= getimage(imageid)
transform_show= T.Resize((h,w))
img_show= transform_show(im)
plt.imshow (img_show)

buffer_f = buffer.view(h, w)
x, y = np.meshgrid(np.arange(w), np.arange(h))
# Flatten x, y, and buffer_f for scatter plot
x, y, sizes = x.flatten(), y.flatten(), buffer_f.flatten()
plt.scatter(x, y, s=sizes*100, c=sizes, cmap='BrBG_r', alpha=0.6)  # Multiplying size for visibility
plt.colorbar(label='Value')


### get the annotation bbox location and map
tensor = getactboxmapver3(h,w,imageid)
x_act, y_act = np.meshgrid(np.arange(w), np.arange(h))
x_act, y_act, sizes = x_act.flatten(), y_act.flatten(), tensor.flatten()
plt.scatter(x_act, y_act, s=sizes * 100, c=sizes, cmap='PRGn', alpha=0.6)



plt.show()

"""
"""
h, w = conv_feature['0'].tensors.shape[-2:]#h = 25 ,w=34
fig, axs = plt.subplots( ncols=len (boxes), nrows=2,figsize=(22,7))
colors = COLORS * 100
for idx, ax_i,(xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T,boxes):


    ax = ax_i[0]
    ax. imshow(dec_attn_weights[0,idx]. view(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}')
    ax = ax_i[1]
    ax.imshow(im)
    ax.add_patch (plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    fill=False, color='blue', linewidth=3))
    ax.axis ('off')
    ax.set_title(coco_classes[str(probas[idx].argmax().item())])
fig.tight_layout()
"""