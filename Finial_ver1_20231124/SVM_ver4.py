#! C:\Users\Vincci\PycharmProjects\detr-main_redownload\venv\Scripts\python.exe
from engine import evaluate
from main import get_args_parser
import torch
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
torch.set_grad_enabled(False)
import json




def rescaled_bbox(out_bbox,size):
    img_w, img_h =size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b*torch.tensor([img_w,img_h, img_w,img_h],dtype=torch.float32)
    return b


def detect(im, model, transform):
    img =transform(im).unsqueeze(0)
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

    return probas[keep],bboxes_scaled,output['pred_logits']

def plot_results (pil_img, prob, boxes):
    plt. figure (figsize=(16 ,10))
    plt. imshow (pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    for p, (xmin, ymin, xmax,ymax),c in zip (prob,boxes.tolist(),colors):

        return [xmin, ymin, xmax,ymax]

with open('./datasets/annotations/instances_val2017.json', 'r') as file:
    text = json.load(file)
categories=text['categories']
coco_id = [item['id'] for item in categories]
coco_data = [item['name'] for item in categories]
coco_classes={f'{coco_id[i]}':coco_data[i] for i in range(len(coco_id))}

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
[0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]





transforms = T.Compose([

    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


model_dump, criterion, postprocessors = build_model(args)



imageid=1503
#1503 31, 61, 71, 92
padded_string = str(imageid).zfill(12) #image id = padding(0)with (12-imagid len) + imageid
image_path=f'.\datasets\\val2017\\{padded_string}.jpg' #39769torch.nonzero(class_detected).flatten()
im=Image.open(image_path)
## call pretrain model from hugconf
detr= detr_resnet101(pretrained=True)


score,boxes,logits=detect(im,detr,transforms)
#print(f"score={score}")
#print(f"logits ={logits}")
#print(f"logits ={logits.shape}")

plot_results(im,score,boxes)





