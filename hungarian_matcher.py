import torch.nn as nn 
import torch  
from box_utils import compute_iou
from scipy.optimize import linear_sum_assignment

# TODO : finish compute_iou methods in box_utils


class HungarianMatcher(nn.Module): 

    def __init__(self, iou_coef, dist_coef): 
        super().__init__() 
        self.iou_coef = iou_coef 
        self.dist_coef = dist_coef


    # we do not want to compute any gradient at this stage, 
    @torch.no_grad()
    def forward(self, y_true, y_pred): 
        """
        Attributes 
        ---------
        y_true : A list of size len(y_true) == bs where each item is a dictionnary 
            {
                "labels" : a list of labels of size number of object in the image 
                "boxes" : a list of boxes of size number of object in the image
            }

        y_pred : A dictionnary of predicted output with two keys 
            {
                "labels" : a tensor of shape (bs, num_queries, num_classes) 
                "boxes" : a tensor of shape (bs, num_queries, 4)
            }


        Output 
        ----------- 
        optimal_indices : list[list[]]  
            A list of size bs of optimal (row, col) pair indices 

        """

        bs, num_queries, num_classes = y_pred["labels"].shape
        device = y_pred["labels"].device
        # get the softmax score and reshape
        y_pred_labels = y_pred["labels"].softmax(-1).view(bs * num_queries, num_classes) # (bs * num_queries, num_classes)
        y_pred_boxes = y_pred["boxes"].view(bs * num_queries, 4) # (bs * num_queries, 4)

        y_true_labels = torch.cat([y["labels"] for y in y_true]) # (bs * num_total_objs) 
        y_true_boxes = torch.cat([y["boxes"] for y in y_true]) # (bs * num_total_objs, 4) 

        # compute probability score 
        # NOTE : the prob_score variable will be a 2D tensor where coordinates (i,j) represent the probability score 
        # between prediction i and ground truth object j, for all images in the batch 

        prob_score = - y_pred_labels[:, y_true_labels] # (bs * num_queries, bs * num_total_objs) 
        
        # L1 distance between true and predictied boxes 
        distance_score = torch.cdist(y_true_boxes, y_pred_boxes, 1) # (bs * num_queries, bs * num_total_objs) 
        # iou score
        IOU_score = compute_iou(y_true_boxes, y_pred_boxes) 

        # compute total score
        match_cost = prob_score + distance_score + IOU_score 
        # reshape for convenience 
        match_cost = match_cost.view(bs, num_queries, -1).cpu() # ((bs, num_queries, bs * num_total_objs)) 

        # get the information regarding how many obejcts are present for each image
        # eg : [1, 4, 10, ...] means that the first image has 1 object, the second 4, the third 10, ... 
        num_objects_per_image = [len(y["bboxes"] for y in y_true)] 

        # we split the cost tensor by the number of images in each image. We split on the last dimension 
        # eg : 
        # (bs, num_queries, 1)
        # (bs, num_queries, 4)
        # (bs, num_queries, 10)
        # ...

        # we access the ith image in the batch by using cost[i] in order to compute the optimal assignement for that image only 
        optimal_matches = [linear_sum_assignment(cost[i]) for i, cost in enumerate(match_cost.split(num_objects_per_image, -1))]
        optimal_indices = [
            (torch.tensor(row_idx, dtype=torch.float32, device=device), torch.tensor(col_idx, dtype=torch.float32, device=device)) for (row_idx, col_idx) in optimal_matches
        ]

        return optimal_indices
    






