import torch 
from criterion import SetCriterion 
from models import DETR
from hungarian_matcher import HungarianMatcher
import unittest



class Tester(unittest.TestCase): 

    def __init__(self, *args, **kwargs): 
        super(Tester, self).__init__(*args, **kwargs)

        self.device = "cuda"
        self.num_queries = 10
        self.d_model = 192
        self.num_patches=7*7
        self.num_head=16
        self.num_encoders=2
        self.num_decoders=2
        self.dropout=0.1
        self.c_out_features=2048
        self.train_backbone=False
        self.num_classes=5
        self.batch_size = 2
        self.input_shape = (self.batch_size, 3, 224, 224) 
        
        self.model = DETR(
            self.num_queries, 
            self.d_model, 
            self.num_patches, 
            self.num_head, 
            self.num_encoders, 
            self.num_decoders, 
            self.dropout, 
            self.c_out_features, 
            self.train_backbone, 
            self.num_classes
        ).to(self.device)
        
        matcher = matcher = HungarianMatcher(1, 1)
        self.criterion = SetCriterion(matcher, self.num_classes)

    @torch.no_grad()
    def test_can_compute_loss(self): 

        # input image 
        img = torch.rand(*self.input_shape, device=self.device, dtype=torch.float32)
        y_true = [{"labels": torch.tensor([1, 3, 2], device=self.device, dtype=torch.float32), "boxes": torch.tensor([[10, 10, 5, 5], [10, 10, 5, 5], [10, 10, 5, 5]], device=self.device, dtype=torch.float32)} for i in range(self.batch_size)]
        output = self.model(img)
        # self.assertTrue(type(output) == dict) 
        # self.assertEqual(tuple(output["labels"].shape), (self.batch_size, self.num_queries, self.num_classes + 1)) 
        # self.assertEqual(tuple(output["boxes"].shape), (self.batch_size, self.num_queries, 4))
        loss = self.criterion(y_true, output)
        self.assertTrue(type(loss), dict) 
        self.assertTrue("label_loss" in loss and "boxes_loss" in loss) 
        self.assertTrue(len(loss["label_loss"].shape) == 0)
        self.assertTrue(len(loss["boxes_loss"].shape) == 0)


if __name__ == '__main__':
    unittest.main()
