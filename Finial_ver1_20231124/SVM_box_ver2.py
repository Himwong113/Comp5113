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
from getcat import coco,bboxlocation,imageid
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
    listbox ={}
    for p, (xmin, ymin, xmax,ymax),c in zip (prob,boxes.tolist(),colors):
        # draw the retangle of box
        ax.add_patch(plt. Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        fill=False, color=c, linewidth=3))
        #p = list all prob in all class
        #cl= p.argmax() find related index about max prog class
        cl = p.argmax()
        print(cl.item())
        ## convert the classs to text , we search the id to find the caterory name
        text = f'{coco(cl.item())}: {p[cl]:0.2f}'
        #print(text)

        key = coco(cl.item())
        value = [xmin, ymin, xmax,ymax]

        if key not in listbox.keys():
            listbox[key]=[]
        listbox[key].append(value)


        ax.text(xmin, ymin, text, fontsize=15,bbox=dict (facecolor= 'yellow', alpha=0.5))
    #plt.axis ('off')
    #plt. show()
    return listbox

def detect2(im, model, transform):
    img =transform(im).unsqueeze(0)
    ## make image to be a tensor
    output = model(img)
    ##generate the prediction
    #print(output['pred_logits'].shape)
    #print(output['pred_logits'].softmax(-1).shape)
    #print(output['pred_logits'].softmax(-1))
    #output['pred_logits'] Shape: (batch_size, num_queries, num_classes)#(1,100,91)


    probasq1 = output['pred_logits'][0,50,:20]


    return probasq1



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


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


model_dump, criterion, postprocessors = build_model(args)



imageids= imageid()
imageids= imageids[:20]
#1503 31, 61, 71, 92
cnt=1

datum = torch.zeros(20)

for idx, imageid in enumerate( imageids):
    padded_string = str(imageid).zfill(12) #image id = padding(0)with (12-imagid len) + imageid
    image_path=f'.\datasets\\val2017\\{padded_string}.jpg' #39769torch.nonzero(class_detected).flatten()
    im=Image.open(image_path)
    ## call pretrain model from hugconf
    detr= detr_resnet101(pretrained=True)
    query = detect2(im, detr, transforms)
    if idx == 0:
        datum=query[idx]
    else:



        for i in range(len(query)):

            plt.scatter(datum,query[i])

plt.show()

#score,boxes,logits=detect(im,detr,transforms)
#print(f"score={score}")
#print(f"logits ={logits}")
#print(f"logits ={logits.shape}")

#pred_bbox = plot_results(im,score,boxes)
#print(pred_bbox)
#act_bbox = bboxlocation(imageid)
#print(act_bbox)







"""
model = detr_resnet50(pretrained=True)
model.eval()

#box_xyxy_to_cxcywh= box_xyxy_to_cxcywh()
device = torch.device(args.device)

coco_val = datasets.coco.build("val", args)
base_ds = get_coco_api_from_dataset(coco_val)

dataset_val = build_dataset(image_set='val', args=args)
utils.init_distributed_mode(args)
if args.distributed:
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)



data_loader_val = DataLoader(
                            dataset_val,
                            args.batch_size,
                            sampler=sampler_val,
                            drop_last=False,
                            collate_fn=utils.collate_fn,
                            num_workers=args.num_workers)




"""