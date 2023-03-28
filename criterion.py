import torch.nn as nn 
import torch 
import torch.nn.functional as F


class SetCriterion(nn.Module):
    """"
        A class that computes the loss between the set of predicted outputs and the ground truth
    """

    def __init__(self, matcher, num_classes): 
        super().__init__() 
        self.matcher = matcher
        self.num_classes = num_classes

    def forward(self, y_true, y_preds): 
        """
            y_true : a list of dictionnary of format {"labels" : [], "boxes": [[], []]}
            y_preds : a dict {"labels": (bs, #Q, num_classes), "boxes" : (bs, #Q, 4)}
        """

        indices = self.matcher(y_true, y_preds)
        label_loss = self.loss_labels(y_preds, y_true, indices)
        boxes_loss = self.loss_boxes(y_preds, y_true, indices)
        
        return {
            "label_loss": label_loss, 
            "boxes_loss": boxes_loss
        }
        

    def loss_labels(self, outputs, targets, indices):
        """
        
        Arguments 
        ----------

        outputs: dict 
            A dictionnary with keys labels and boxes with respectively the shapes (bs, #Q, num_classes) and (bs, #Q, 4)
        targets: list[dict]
            A list of size batch_size of dict with keys labels and boxes. The values of each dictionnary are lists containing labels and boxes 
        indices: tuple[list[], list[]]
            A tuple with two lists. The first list contains the indices of the predicted query boxes, the second list contains the indices of the ground truth boxes
            and (i,j) represents the relationship between predicted query i and ground truth j 
        """

        src_logits = outputs['labels'] # (bs, #Q, 92)
        idx = self._get_src_permutation_idx(indices) # ([number_of_objects], [number_of_objects])
        # get all label from the batch
        target_classes_o = torch.cat([t["labels"][J.type(torch.int32)] for t, (_, J) in zip(targets, indices)])
        # create target classes with only no object classes, with same shape as the predicted labels 
        target_classes = torch.full((int(src_logits.shape[0]), int(src_logits.shape[1])), self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) # (bs, #Q)
        target_classes = target_classes.type(torch.float32)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes.long())
        return loss_ce
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # batch_ids assigns a query to a batch
        batch_idx = torch.cat([torch.full_like(src, i, dtype=torch.int32) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices]).type(torch.int32)
        return batch_idx, src_idx

    def loss_boxes(self, outputs, targets, indices): 
        """
        
        Arguments 
        ----------

        outputs: dict 
            A dictionnary with keys labels and boxes with respectively the shapes (bs, #Q, num_classes) and (bs, #Q, 4)
        targets: list[dict]
            A list of size batch_size of dict with keys labels and boxes. The values of each dictionnary are lists containing labels and boxes 
        indices: tuple[list[], list[]]
            A tuple with two lists. The first list contains the indices of the predicted query boxes, the second list contains the indices of the ground truth boxes
            and (i,j) represents the relationship between predicted query i and ground truth j 
        """ 

        # retrieve boxes from prediction
        pred_boxes = outputs["boxes"] # (bs, #Q, 4)

        # retrieve ground truth boxes 
        idx = self._get_src_permutation_idx(indices) # (batch_size*num_obj), (batch_size*num_obj)
        target_boxes = torch.cat([t["boxes"][j] for t, (_,j) in zip(targets, indices)])
        pred_boxes = pred_boxes[idx]

        l1_loss = F.l1_loss(target_boxes, pred_boxes) 

        return l1_loss




