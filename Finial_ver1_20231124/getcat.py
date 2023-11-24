
"""
import torch
print(torch.backends.cudnn.enabled)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())

"""

import json
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T


def coco(id=None):
    with open('./datasets/annotations/instances_val2017.json', 'r') as file:
        text = json.load(file)

    categories=text['categories']
    coco_id = [item['id'] for item in categories]
    coco_data = [item['name'] for item in categories]

    coco_set={f'{coco_id[i]}':coco_data[i] for i in range(len(coco_id))}

    existing_keys = set(coco_set.keys())
    missing_keys = [str(i) for i in range(1,90) if str(i) not in existing_keys]


    for missing_key in missing_keys:
        coco_set[missing_key] ='N\A'

    coco_set['91']= 'no object'
    if id is not None:
        id = str(id)
        return coco_set[id]
    else:
        return coco_set

def bboxlocation(id):
    with open('./datasets/annotations/instances_val2017.json', 'r') as file:
        text = json.load(file)

    annotations =text['annotations']
    box = [item["bbox"]for item in annotations]
    image_id = [item["image_id"]for item in annotations]
    category_id = [item["category_id"] for item in annotations]
    boxset = defaultdict(list)
    info= defaultdict(list)

    for i in range(len(box)):
        image_id_i = image_id[i]
        category_id_i = category_id[i]
        bbox_i = box[i]

        # Append a tuple containing category_id and bbox to the list
        boxset[image_id_i].append((category_id_i, bbox_i))

    listbox = {}
    for item in boxset[id]:
        if coco(item[0]) not in listbox.keys():
            listbox[coco(item[0])] = []

        listbox[coco(item[0])].append(item[1])

    return listbox


def imageid():
    with open('./datasets/annotations/instances_val2017.json', 'r') as file:
        text = json.load(file)

    annotations =text['annotations']
    image_id = [item["image_id"]for item in annotations]




    return image_id




def center_bbox(act_bbox):
    center_bbox = {}
    for k in act_bbox:
        for v in act_bbox[k]:

            if k not in center_bbox.keys():
                center_bbox[k]=[]

            center_bbox[k].append( [ (v[0]+v[2])/2 ,(v[1]+v[3])/2])
    return center_bbox



def getactboxmap(h,w,imageid):
    actualbbox= bboxlocation(imageid)

    padded_string = str(imageid).zfill(12)  # image id = padding(0)with (12-imagid len) + imageid
    image_path = f'.\datasets\\val2017\\{padded_string}.jpg'  # 39769torch.nonzero(class_detected).flatten()
    im = Image.open(image_path)
    img_w, img_h= im.size



    tensor = torch.zeros((h, w))  # Create a tensor of zeros

    for k in actualbbox:
        for v in actualbbox[k]:




            x_tl = v[0] if int(v[0] )!= 0 else 1
            y_tl = v[1] if int(v[1])!= 0 else 1
            ww =  v[2] if int(v[2] )!= 0 else 1
            hh =  v[3] if int(v[3])!= 0 else 1

            x_c =  x_tl    + 0.5 * ww
            y_c =  y_tl    - 0.5 * hh

            """
            xmin,  ymin,xmax,  ymax =[int((x_c - 0.5 * ww)/img_w*w),
                                      int((y_c - 0.5 * hh)/img_h*h),
                                      int((x_c + 0.5 * ww)/img_w*w),
                                      int((y_c + 0.5 * hh)/img_h*h)]
            """
            xmax,ymin,ymax,xmin = [     int((x_c - 0.5 * ww) / img_w * w),
                                        int((y_c - 0.5 * hh) / img_h * h),
                                        int((x_c + 0.5 * ww) / img_w * w),
                                        int((y_c + 0.5 * hh) / img_h * h)]










            # Set the values in the specified rectangle area to 1
            tensor[ymin:ymax+1,xmin:xmax+1] = 1


    return tensor

def getfinialunshapeboxmap(imageid):
    actualbbox= bboxlocation(imageid)

    padded_string = str(imageid).zfill(12)  # image id = padding(0)with (12-imagid len) + imageid
    image_path = f'.\datasets\\val2017\\{padded_string}.jpg'  # 39769torch.nonzero(class_detected).flatten()
    im = Image.open(image_path)
    img_w, img_h= im.size
    print( img_w, img_h)



    tensor = torch.zeros((img_h, img_w))  # Create a tensor of zeros

    for k in actualbbox:
        for v in actualbbox[k]:




            x_tl = v[0]
            y_tl = v[1]
            ww =  v[2] 
            hh =  v[3]

            x_c =  x_tl    + 0.5 * ww
            y_c =  y_tl    + 0.5 * hh
            xmin,  ymin,xmax,  ymax =[int(x_c - 0.5 * ww),
                                      int(y_c - 0.5 * hh),
                                      int(x_c + 0.5 * ww),
                                      int(y_c + 0.5 * hh)]

            xmin= xmin if int(xmin) > 0 else 1
            xmax= xmax if int(xmax) > 0 else 1
            ymin= ymin if int(ymin) > 0 else 1
            ymax= ymax if int(ymax) > 0 else 1

            print(f'xmin,xmax,  ymin, ymax={xmin,xmax,  ymin, ymax}')


            # Set the values in the specified rectangle area to 1
            tensor[ymin:ymax +1,xmin:xmax+1] = 1


    return tensor


def rotate_rectangle(xmin, xmax, ymin, ymax, theta_deg,x_c,y_c ):
    """
    Rotate a rectangle defined by its corner points (xmin, xmax, ymin, ymax) around
    a center point by theta degrees clockwise and return the coordinates of the
    rotated rectangle.

    Parameters:
    - xmin, xmax, ymin, ymax: coordinates of the original rectangle
    - theta_deg: angle in degrees to rotate the rectangle clockwise
    - center: a tuple (x_center, y_center) representing the center of rotation

    Returns:
    - rotated_coords: coordinates of the rotated rectangle as (xmin, xmax, ymin, ymax)
    """
    center = (x_c, y_c)
    # Convert theta from degrees to radians
    theta_rad = np.radians(theta_deg)

    # Rotation matrix for clockwise rotation
    rotation_matrix = np.array([[np.cos(theta_rad), np.sin(theta_rad)],
                                [-np.sin(theta_rad), np.cos(theta_rad)]])

    # Original corner points
    points = np.array([[xmin, ymin],
                       [xmin, ymax],
                       [xmax, ymax],
                       [xmax, ymin]])

    # Translate points to origin for rotation
    translated_points = points - center

    # Apply rotation
    rotated_points = np.dot(translated_points, rotation_matrix)

    # Translate points back from origin
    rotated_points += center

    # Get the min and max x and y from the rotated points
    xmin_rotated, ymin_rotated = np.min(rotated_points, axis=0)
    xmax_rotated, ymax_rotated = np.max(rotated_points, axis=0)

    # Return the coordinates as a tuple
    rotated_coords = (xmin_rotated, xmax_rotated, ymin_rotated, ymax_rotated)

    return xmin_rotated, xmax_rotated, ymin_rotated, ymax_rotated


def getactboxmapver2(h,w,imageid):
    actualbbox= bboxlocation(imageid)

    padded_string = str(imageid).zfill(12)  # image id = padding(0)with (12-imagid len) + imageid
    image_path = f'.\datasets\\val2017\\{padded_string}.jpg'  # 39769torch.nonzero(class_detected).flatten()
    im = Image.open(image_path)
    img_h,img_w= im.size



    tensor = torch.zeros((h,w))  # Create a tensor of zeros

    for k in actualbbox:
        for v in actualbbox[k]:
            x_tl= v[0] if int(v[0]) > 0 else 1
            y_tl= v[1] if int(v[1]) > 0 else 1
            ww=   v[2] if int(v[2]) > 0 else 1
            hh=   v[3] if int(v[3]) > 0 else 1


            x_c = x_tl + 0.5 * ww   
            y_c = y_tl - 0.5 * hh


            print(f' x_c ,y_c,ww ,hh ={ x_c ,y_c,ww ,hh}')
            xmin,  ymin,xmax,  ymax =[int(x_c - 0.5 * ww),
                                      int(y_c - 0.5 * hh),
                                      int(x_c + 0.5 * ww),
                                      int(y_c + 0.5 * hh)]


            print(f'{ymax}={y_c}+{0.5 }* {hh}')

            #xmin, xmax, ymin, ymax=rotate_rectangle(xmin, xmax, ymin, ymax, -90,xmin,ymax )
            print(f' xmin, xmax, ymin, ymax ={xmin, xmax, ymin, ymax }')
            """
             # Calculate bounding box coordinates based on image and target dimensions
            xmin = int((x_c - 0.5 * ww) * w / img_w)
            ymin = int(((y_c - 0.5 * hh)) * h / img_h)
            xmax = int((x_c + 0.5 * ww) * w / img_w)
            ymax = int(((y_c + 0.5 * hh)) * h / img_h)
            print(f' xmax,ymin,ymax,xmin ={ xmax,ymin,ymax,xmin }')
            
            xmin = int(xmin *(max(w,h)/min(w,h))* w / img_w)  if h>w else   int(xmin *w / img_w)
            ymin = int(ymin *(max(w,h)/min(w,h))* h / img_h)  if w>h else       int(ymin *h / img_h)
            xmax = int(xmax *(max(w,h)/min(w,h))* w / img_w)  if h>w else       int(xmax *w / img_w)
            ymax = int(ymax *(max(w,h)/min(w,h))* h / img_h)  if w>h else       int(ymax *h / img_h)
            xmin = int((x_c - 0.5 * ww)     * w )                   
            ymin = int(((y_c - 0.5 * hh))   * h )                   
            xmax = int((x_c + 0.5 * ww)     * w )                   
            ymax = int(((y_c + 0.5 * hh))   * h )                   
            print(f' xmax,ymin,ymax,xmin ={xmax, ymin, ymax, xmin}')


            xmin = max(min(xmin, w - 1), 0)
            xmax = max(min(xmax, w - 1), 0)
            ymin = max(min(ymin, h - 1), 0)
            ymax = max(min(ymax, h - 1), 0)
            """
            xmin_ratio =(xmin)/ (img_w)
            xmax_ratio =(xmax)/ (img_w)
            ymin_ratio =(ymin)/ (img_h)
            ymax_ratio =(ymax)/ (img_h)
            print(f' xmin, xmax, ymin, ymax ={xmin_ratio, xmax_ratio, ymin_ratio, ymax_ratio }')
            xmin = int( w*xmin_ratio)
            xmax = int( w*xmax_ratio)
            ymin = int( h*ymin_ratio)
            ymax = int( h*ymax_ratio)
            print(f' xmin, xmax, ymin, ymax ={xmin, xmax, ymin, ymax }')
            xmin = max(xmin, 0)
            xmax = min(xmax, w)
            ymin = max(ymin, 0)
            ymax = min(ymax, h)



            print(f' xmin, xmax, ymin, ymax ={xmin, xmax, ymin, ymax }')
            # Ensure that coordinates are within the tensor dimensions




            tensor[ymin:ymax + 1, xmin:xmax + 1] = 1

        # Resize the final tensor to the desired dimensions
    print(tensor.shape)
    #resized_tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
    #resized_tensor = resized_tensor.squeeze()



    return tensor


def getactboxmapver3(h,w,imageid):
    actualbbox = bboxlocation(imageid)

    padded_string = str(imageid).zfill(12)  # image id = padding(0)with (12-imagid len) + imageid
    image_path = f'.\datasets\\val2017\\{padded_string}.jpg'  # 39769torch.nonzero(class_detected).flatten()
    im = Image.open(image_path)
    img_w, img_h = im.size


    tensor = torch.zeros((img_h, img_w))  # Create a tensor of zeros

    for k in actualbbox:
        for v in actualbbox[k]:
            x_tl = v[0]
            y_tl = v[1]
            ww = v[2]
            hh = v[3]

            x_c = x_tl + 0.5 * ww
            y_c = y_tl + 0.5 * hh
            xmin, ymin, xmax, ymax = [int(x_c - 0.5 * ww),
                                      int(y_c - 0.5 * hh),
                                      int(x_c + 0.5 * ww),
                                      int(y_c + 0.5 * hh)]

            xmin = xmin if int(xmin) > 0 else 1
            xmax = xmax if int(xmax) > 0 else 1
            ymin = ymin if int(ymin) > 0 else 1
            ymax = ymax if int(ymax) > 0 else 1

            #print(f'xmin,xmax,  ymin, ymax={xmin, xmax, ymin, ymax}')

            # Set the values in the specified rectangle area to 1
            tensor[ymin:ymax + 1, xmin:xmax + 1] = 1

    resized_tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
    resized_tensor = resized_tensor.squeeze()
    return resized_tensor





"""

imageid=1503
padded_string = str(imageid).zfill(12)
image_path=f'.\datasets\\val2017\\{padded_string}.jpg' #39769torch.nonzero(class_detected).flatten()
im=Image.open(image_path)
w,h= 35,24
transform_show= T.Resize((h,w))
img_show= transform_show(im)
plt.imshow (img_show)


tensor = getactboxmapver3(h,w,imageid)

x_act, y_act = np.meshgrid(np.arange(w), np.arange(h))
x_act, y_act, sizes = x_act.flatten(), y_act.flatten(), tensor.flatten()
plt.scatter(x_act, y_act, s=sizes * 100, c=sizes, cmap='PRGn', alpha=0.6)

plt.show()
h,w,imageid=25,34,1503
tensor = getactboxmap(h,w,imageid)
x_act, y_act = np.meshgrid(np.arange(w), np.arange(h))
x_act, y_act, sizes = x_act.flatten(), y_act.flatten(), tensor.flatten()
plt.scatter(x_act, y_act, s=sizes * 100, c=sizes, cmap='PRGn', alpha=0.6)

plt.show()

"""




"""
x_act, y_act = np.meshgrid(np.arange(w), np.arange(h))
x_act, y_act, sizes = x_act.flatten(), y_act.flatten(), tensor.flatten()
plt.scatter(x_act, y_act, s=sizes * 100, c=sizes, cmap='PRGn', alpha=0.6)


plt.show()


"""

