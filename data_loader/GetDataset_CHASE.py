# -*- coding: UTF-8 -*-

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
import random
import os
import scipy.io as scio
from skimage.io import imread, imsave
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision.transforms.functional as TF
import scipy.misc as misc

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask


def default_loader(img_path, mask_path):

    img = cv2.imread(img_path)
    # print("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (448, 448))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (448, 448))
    
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    
    mask = np.expand_dims(mask, axis=2)
    #
    # print(np.shape(img))
    # print(np.shape(mask))

    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    #mask = abs(mask-1)
    return img, mask

def default_DRIVE_loader(img_path, mask_path, train=True):
    try:
        # Read image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            raise ValueError(f"Failed to load image or mask: {img_path}")
        
        # Resize to 480x480 to match model output
        img = cv2.resize(img, (480, 480))
        mask = cv2.resize(mask, (480, 480))
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0
        
        # Convert to tensor
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        mask = torch.from_numpy(mask[None, :, :])
        
        return img, mask
    except Exception as e:
        print(f"Error loading {img_path}: {str(e)}")
        raise

class Normalize(object):
    def __call__(self, image, mask=None):
        # image = (image - self.mean)/self.std

        image = (image-image.min())/(image.max()-image.min())
        mask = mask/255.0
        if mask is None:
            return image
        return image, mask

class RandomCrop(object):
    def __call__(self, image, mask=None):
        H,W   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        return image[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1].copy(), mask[:,::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask

def Resize(image, mask,H,W):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask  = cv2.resize( mask, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        return image, mask
    else:
        return image

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)

        return image, mask

def _resize_image(image, target):
   return cv2.resize(image, dsize=(target[0], target[1]), interpolation=cv2.INTER_LINEAR)


#root = '/home/ziyun/Desktop/Project/Mice_seg/Data_train'
class MyDataset_CHASE(data.Dataset):# 
    def __init__(self, args, train_root, pat_ls, mode='train'): 
        self.args = args
        self.train = (mode == 'train')
        self.root = train_root
        
        # Initialize lists
        self.img_ls = []
        self.mask_ls = []
        
        # Directly look for files in the root directory
        for i in range(1, 15):  # CHASEDB1 has 14 images
            for side in ['L', 'R']:
                img_name = f'Image_{i:02d}{side}.jpg'  # Image file
                mask_name = f'Image_{i:02d}{side}_1stHO.png'  # Mask file
                
                img_path = os.path.join(self.root, img_name)
                mask_path = os.path.join(self.root, mask_name)
                
                # Check if both the image and mask exist
                if os.path.isfile(img_path) and os.path.isfile(mask_path):
                    self.img_ls.append(img_path)
                    self.mask_ls.append(mask_path)
        
        if len(self.img_ls) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {self.root}")
            
        print(f"Found {len(self.img_ls)} image-mask pairs for {mode} mode.")

        # Data augmentation and preprocessing
        self.normalize  = Normalize()
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.totensor   = ToTensor()

    def __getitem__(self, index):
        # Load image and mask
        img, mask = default_DRIVE_loader(self.img_ls[index], self.mask_ls[index], self.train)

        # Convert to tensors
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        # Return only the image and mask
        return img.squeeze(0), mask

    def __len__(self):
        return len(self.img_ls)


def connectivity_matrix(mask):
    # print(mask.shape)
    [batch,channels,rows, cols] = mask.shape

    conn = torch.ones([batch,8,rows, cols])
    up = torch.zeros([batch,rows, cols])#move the orignal mask to up
    down = torch.zeros([batch,rows, cols])
    left = torch.zeros([batch,rows, cols])
    right = torch.zeros([batch,rows, cols])
    up_left = torch.zeros([batch,rows, cols])
    up_right = torch.zeros([batch,rows, cols])
    down_left = torch.zeros([batch,rows, cols])
    down_right = torch.zeros([batch,rows, cols])


    up[:,:rows-1, :] = mask[:,0,1:rows,:]
    down[:,1:rows,:] = mask[:,0,0:rows-1,:]
    left[:,:,:cols-1] = mask[:,0,:,1:cols]
    right[:,:,1:cols] = mask[:,0,:,:cols-1]
    up_left[:,0:rows-1,0:cols-1] = mask[:,0,1:rows,1:cols]
    up_right[:,0:rows-1,1:cols] = mask[:,0,1:rows,0:cols-1]
    down_left[:,1:rows,0:cols-1] = mask[:,0,0:rows-1,1:cols]
    down_right[:,1:rows,1:cols] = mask[:,0,0:rows-1,0:cols-1]

    # print(mask.shape,down_right.shape)
    conn[:,0] = mask[:,0]*down_right
    conn[:,1] = mask[:,0]*down
    conn[:,2] = mask[:,0]*down_left
    conn[:,3] = mask[:,0]*right
    conn[:,4] = mask[:,0]*left
    conn[:,5] = mask[:,0]*up_right
    conn[:,6] = mask[:,0]*up
    conn[:,7] = mask[:,0]*up_left
    conn = conn.float()

    return conn



def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def check_label(mask):
    label = np.array([1,0,0,0])
    # print(mask.shape)
    # print(mask[1,:,:].max())
    if mask[1,:,:].max()!=0:
        label[1]=1

    if mask[2,:,:].max()!=0:
        label[2]=1

    if mask[3,:,:].max()!=0:
        label[3]=1

    return label

# def thres_multilabel(mask):
#     mask[np.where(mask<0.5)]=0
#     mask[np.where((mask<1.5) & (mask>=0.5))]=1
#     mask[np.where((mask<2.5) & (mask>=1.5))]=2
#     mask[np.where(mask>2.5)]=3

#     return mask

def verify_chase_dataset(data_root):
    """Verify the dataset structure and print helpful debug information"""
    images_dir = os.path.join(data_root, 'images')
    masks_dir = os.path.join(data_root, 'masks')
    
    print("\nVerifying CHASE dataset structure:")
    print(f"Looking in: {data_root}")
    
    if os.path.exists(images_dir):
        print(f"\nImages directory exists at: {images_dir}")
        image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
        print(f"Found {len(image_files)} jpg files")
        if len(image_files) > 0:
            print("Sample image files:", image_files[:3])
    else:
        raise RuntimeError(f"Images directory not found at: {images_dir}")
    
    if os.path.exists(masks_dir):
        print(f"\nMasks directory exists at: {masks_dir}")
        mask_files = glob.glob(os.path.join(masks_dir, '*_1stHO.png'))
        print(f"Found {len(mask_files)} mask files")
        if len(mask_files) > 0:
            print("Sample mask files:", mask_files[:3])
    else:
        raise RuntimeError(f"Masks directory not found at: {masks_dir}")