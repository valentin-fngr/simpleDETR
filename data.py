from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
import torch 
from torch.utils.data import DataLoader
import config 


data_dir = 'coco'

# Define the transforms to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize the images to 224x224
    transforms.ToTensor(),          # convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the images
])

# Load the COCO dataset using the CocoDetection class
coco_train = CocoDetection(root=data_dir, annFile=data_dir+'/annotations/instances_train2017.json', transform=transform)
coco_val = CocoDetection(root=data_dir, annFile=data_dir+'/annotations/instances_val2017.json', transform=transform)

# Print the number of images and categories in the train and validation sets
print(f'Number of train images: {len(coco_train)}')
print(f'Number of validation images: {len(coco_val)}')
print(f'Number of categories: {len(coco_train.coco.cats)}')