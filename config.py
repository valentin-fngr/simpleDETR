# hyper parameters 

train_split = 0.8 
batch_size = 16



# pre process 
image_size = 224
num_patches = 8
if image_size % num_patches != 0: 
    raise ValueError("image size not divisble by num_patches, ", (image_size, num_patches))

