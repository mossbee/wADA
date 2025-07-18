import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
import torch

# Global MTCNN model - will be initialized when needed
mtcnn_model = None

def get_mtcnn_model(device=None):
    """Get or create MTCNN model with specified device"""
    global mtcnn_model
    if mtcnn_model is None or (device is not None and str(mtcnn_model.device) != device):
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        mtcnn_model = mtcnn.MTCNN(device=device, crop_size=(112, 112))
    return mtcnn_model

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None, device=None):
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    # find face
    try:
        mtcnn_model = get_mtcnn_model(device)
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0]
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        face = None

    return face


