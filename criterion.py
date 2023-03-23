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
        return label_loss
        

    def loss_labels(self, outputs, targets, indices):

        src_logits = outputs['labels'] # (bs, #Q, 92)
        print("how many images in the batch : ", len(targets))
        print("How many total objects : ", sum([len(t["labels"]) for t in targets]))

        idx = self._get_src_permutation_idx(indices) # ([number_of_objects], [number_of_objects])
        # get all label from the batch
        target_classes_o = torch.cat([t["labels"][J.type(torch.int32)] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes + 1,
                                    dtype=torch.int64, device=src_logits.device) # (bs, #Q)
        target_classes = target_classes.type(torch.float32)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes.long())
        return loss_ce
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i, dtype=torch.int32) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices]).type(torch.int32)
        return batch_idx, src_idx


