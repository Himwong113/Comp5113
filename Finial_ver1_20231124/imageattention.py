
import torch

from hubconf import detr_resnet50,_make_detr,detr_resnet101
from util.box_ops import box_xyxy_to_cxcywh,box_cxcywh_to_xyxy
import torchvision.transforms as T

import matplotlib.pyplot as plt
from imageget import getimage
import json

torch.set_grad_enabled(False)

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

#1503 31, 61, 71, 92
def endattnvisl (imageid):
    im=getimage(imageid)
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
    plt.show()



########
def decodeMLHvisl(imageid):
    im = getimage(imageid)
    ## call pretrain model from hugconf
    detr = detr_resnet101(pretrained=True)
    score, boxes, logits, keep, probas = detect(im, detr, transforms)
    ##### encoder attention
    conv_feature, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        detr.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_feature.append(output)
        ),
        detr.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
            # grab the last layer in encoder from MLA #ie take the attention weighted
        ),
        detr.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
            # take the MHA layer which have qkv form encode and qkv from decoder # ie 2nd MHA in endcoder
        )
    ]

    img = transforms(im).unsqueeze(0)
    output = detr(img)

    for hook in hooks:
        hook.remove()
    conv_feature = conv_feature[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]


    img = transforms(im).unsqueeze(0)
    #### decoder attention
    f_map = conv_feature['0']
    shape = f_map.tensors.shape[-2:]
    sattn = enc_attn_weights[0].reshape(shape+shape)
    #print(sattn.shape)

    fact = 32
    # let's select 4 reference points for visualization
    ids = [(200, 200), (280, 400), (200, 600), (440, 800),]
    #here we create the canvas
    fig = plt. figure (constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot (gs[0, 0]),
        fig.add_subplot (gs[1, 0]),
        fig.add_subplot (gs[0, -1]),
        fig.add_subplot (gs[1, -1]),

    ]

    for idx_o, ax in zip(ids, axs):
        idx = (idx_o[0] // fact, idx_o[1]//fact)
        ax. imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation= 'nearest')
        ax.axis ('off')
        ax.set_title(f"self-attention{idx_o}")

    fcenter_ax =fig.add_subplot(gs[:,1:-1])
    fcenter_ax.imshow(im)
    for (y, x) in ids:
        scale = im.height /img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
        fcenter_ax.axis ('off')

    plt.show()

