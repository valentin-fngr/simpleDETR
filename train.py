from models import DETR
from criterion import SetCriterion 
from hungarian_matcher import HungarianMatcher 
from data import COCODataset
import config 
from torch.utils.data import DataLoader
import torch.optim as optim 
from tqdm import tqdm 


def get_data(): 

    train = COCODataset(config.image_train, config.annot_train, config.image_size, config.image_size, config.num_classes)
    val = COCODataset(config.image_val, config.annot_val, config.image_size, config.image_size, config.num_classes)

    train_loader = DataLoader(train, batch_size=config.batch_size)
    val_loader = DataLoader(val, batch_size=config.batch_size)

    return train_loader, val_loader


def get_model(): 

    model = DETR(
        config.num_queries,
        config.d_model, 
        config.num_patches, 
        config.num_head, 
        config.num_encoders, 
        config.num_decoders, 
        config.dropout
    )

    model = model.to(config.device) 
    return model 


def get_criterion(): 
    matcher = HungarianMatcher(1, 1)
    criterion = SetCriterion(matcher, config.num_classes)
    return criterion 


def get_optimizer(model): 

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    return optimizer 



def train(model, train_loader, criterion, optimizer, epoch, writer=None): 

    model.train()

    total_loss = 0.0 
    total_boxes_loss = 0.0 
    total_labels_loss = 0.0

    for i, data in enumerate(tqdm(train_loader)): 

        images, targets = data 
        images = images.to(config.device) 

        # this intellectual gymnastic is only necessary because I decided to follow 
        # the format of each inputs and outputs of the official implementation
        targets_labels = targets["labels"] 
        targets_boxes = targets["boxes"]
        targets = [{"labels": labels.to(config.device), "boxes": boxes.to(config.device)}for labels, boxes in zip(targets_labels, targets_boxes)]

        preds = model(images) 
        # compute loss 
        losses = criterion(targets, preds)
        label_loss = losses["label_loss"]
        boxes_loss = losses["boxes_loss"]

        final_loss = label_loss + boxes_loss

        total_loss += final_loss.item() 
        total_boxes_loss += boxes_loss.item() 
        total_labels_loss += label_loss.item()

        if i % 200 == 0: 
            print(f"Epoch {epoch} [{i}|{len(train_loader)}] : total_loss={total_loss/(i+1)} _ total_boxes_loss={total_boxes_loss/(i+1)} _ total_labels_loss={total_labels_loss/(i+1)}")            

        model.zero_grad()
        final_loss.backward() 
        optimizer.step()

    total_loss /= len(train_loader)
    total_boxes_loss /= len(train_loader)
    total_labels_loss /= len(train_loader)

    if writer is not None: 
        
        writer.add_scalar("Train/total_loss", total_loss, epoch) 
        writer.add_scalar("Train/total_boxes_loss", total_boxes_loss, epoch) 
        writer.add_scalar("Train/ttoal_labels_loss", total_labels_loss, epoch) 


    return 



def main(): 

    train_loader, val_loader = get_data()
    criterion = get_criterion() 
    model = get_model() 
    optimizer = get_optimizer(model)

    for epoch in range(config.epochs): 

        train(model, train_loader, criterion, optimizer, epoch, writer=None)

    


    return 









if __name__ == "__main__": 
    
    main()