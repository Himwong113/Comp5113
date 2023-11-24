from PIL import Image
import matplotlib.pyplot as plt

def getimage(imageid , path=None ,type=None):
    padded_string = str(imageid).zfill(12)  # image id = padding(0)with (12-imagid len) + imageid
    if path == None and type ==None:
        image_path = f'.\datasets\\val2017\\{padded_string}.jpg'  # 39769torch.nonzero(class_detected).flatten()
    elif(path == None and type == 'train'):
        image_path = f'.\datasets\\train2017\\{padded_string}.jpg'

    else:
        image_path = path+f'{padded_string}.jpg'

    im = Image.open(image_path)
    return im
