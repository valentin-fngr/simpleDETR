import torch  




def from_cxcy_to_tlxtly(boxes): 
    """
    Convert the format of boxes from center to top left. 

    Attributes 
    -----------

    boxes : tensor (P, 4)
        boxes with center style format (centerX, centerY, w, h)

    Output 
    ----------
    _boxes : tensor (P, 4)
        boxes with corner style format
    """

    _boxes = boxes.clone()
    _boxes[:, 0], _boxes[:, 1] = (_boxes[:, 0] - _boxes[:, 2] * 0.5), (_boxes[:, 1] - _boxes[:, 3] * 0.5)
    _boxes[:, 2], _boxes[:, 3] = _boxes[:, 0] + _boxes[:, 2], _boxes[:, 1] + _boxes[:, 3]
    return _boxes 





def compute_iou(boxes1, boxes2): 
    """
    Compute the IOU matrix between boxes1 and boxes2.
    All boxes must be with format (tl_x, tl_y, br_x, br_y).
    
    Attributes 
    ---------- 

    boxes1 : tensor (P, 4) 
        a set of P boxes 
    
    boxes2 : tensor (N, 4) 
        a set of N boxes


    Output 
    ---------- 
    iou : tensor (P, N)
        iou matrix between boxes1 and boxes2
    """
    # get intersection coordinates
    top_left = torch.maximum(boxes1[None, :, :2], boxes2[:, None, :2]) # (P, 1, 2)
    bottom_right = torch.minimum(boxes1[None, :, :2], boxes2[:, None, :2]) # (N, 1, 2)

    inter_wh = (bottom_right - top_left)
    inter_area = inter_wh[:, : , 0] * inter_wh[:, : , 1]

    area1 = (boxes1[:, 2:] - boxes1[:, :2])
    area1 = area1[:, 0] * area1[:, 1]
    area2 = (boxes2[:, 2:] - boxes2[:, :2])
    area2 = area2[:, 0] * area2[:, 1] 

    union = area1[None, :] + area2[:, None] - inter_area 
    iou = torch.clip(inter_area / union, min=0) 
    return iou 




if __name__ == "__main__": 

    boxes1 = [[10, 10, 5, 5], [10.4, 10.4, 5, 5]]
    boxes1 = torch.tensor(boxes1, dtype=torch.float32) 
    # print("old : ", boxes1)
    # new_boxes = from_cxcy_to_tlxtly(boxes1)
    # print("new : ", new_boxes)


    print("compute IOU : ") 
    iou_matrix = compute_iou(boxes1, boxes1)
    print(iou_matrix)