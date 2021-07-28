import os, random
import argparse
import numpy as np
import torch
import torch.optim as optim
import timm.models as tm_models
import pretrainedmodels as pt_models
import models
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import cv2
import csv

test_root = 'data/test/'
gpu = 1
name = 'efficientnet_b4'
ckpt = 'saved/%s.pt' % name

test_transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# model = tm_models.create_model(model, num_classes=137)
model = torch.load(ckpt, map_location=torch.device('cpu'))
model = model.cuda(gpu)
model.eval()
test_images = tqdm(os.listdir(test_root), ncols=100)

classes = [d.name for d in os.scandir('data/train') if d.is_dir()]
classes.sort()

with open('res/res-%s.csv'%name, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'category_id'])
    for im in test_images:
        # img = cv2.imread(os.path.join(test_root, i), cv2.IMREAD_ANYCOLOR)  
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = Image.open(os.path.join(test_root, im)).convert("RGB")
        img = test_transform(img).unsqueeze(0)
        img = img.cuda(gpu)
        output = model(img)
        label = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        writer.writerow([im, classes[label]])