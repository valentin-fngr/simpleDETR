from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import torch 
from torch.utils.data import DataLoader, Dataset
import config 
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np 
import config

data_dir = 'coco'

# Load the COCO dataset using the CocoDetection class
class COCODataset(Dataset):
    def __init__(self, root: str, annotation: str, targetHeight: int, targetWidth: int, numClass: int):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())

        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.numClass = numClass

        self.transforms = T.Compose([
            # T.RandomOrder([
            #     T.RandomHorizontalFlip(),
            #     T.RandomSizeCrop(numClass)
            # ]),
            T.ToTensor(),
            T.Resize((targetHeight, targetWidth)),
            T.ColorJitter(brightness=.2, contrast=.1, saturation=.1, hue=0),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

        with open('classes.txt', 'w') as f:
            f.write(str(classes))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        """
        Return
            image (3, H, W)
            target: {
                "boxes" : (nb_boxes, 4), 
                "labels" : (nb_boxes, )
            }

            Note that boxes are formated such as : (centerX, centerY, w, h) + rescaled to image
        """
        imgID = self.ids[idx]

        imgInfo = self.coco.imgs[imgID]
        imgPath = os.path.join(self.root, imgInfo['file_name'])

        image = Image.open(imgPath).convert('RGB')
        annotations = self.loadAnnotations(imgID, imgInfo['width'], imgInfo['height'])
        if len(annotations) == 0:
            print("annotations empty ! ")
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),
            }
            print(targets)
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),
            }

        image = self.transforms(image)

        return image, targets

    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []

        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']

            # convert from [tlX, tlY, w, h] to [centerX, centerY, w, h]
            bbox[0] += bbox[2] / 2
            bbox[1] += bbox[3] / 2

            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]

            ans.append(bbox + [cat])

        return np.asarray(ans)
    