# hyper parameters 
import torch

train_split = 0.8 
batch_size = 128
lr = 10e-4
epochs = 50 
dropout = 0.1

# pre process 
image_size = 224
num_patches = 49

# classes number 91 is noobj
num_classes = 91
num_queries = 60
d_model = 192
num_head = 6
num_encoders = num_decoders = 3


image_train = "coco" + "/images/train2017"
annot_train = "coco"+"/annotations/instances_train2017.json"
image_val = "coco" + "/images/val2017"
annot_val = "coco" + "/annotations/instances_val2017.json"


# training 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')