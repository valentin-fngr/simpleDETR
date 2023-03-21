# hyper parameters 

train_split = 0.8 
batch_size = 16



# pre process 
image_size = 224

# classes number 91 is noobj
num_classes = 91


image_train = "coco" + "/images/train2017"
annot_train = "coco"+"/annotations/instances_train2017.json"
image_val = "coco" + "/images/val2017"
annot_val = "coco" + "/annotations/instances_val2017.json"