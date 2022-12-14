import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset.datasets import TrainDataset, ValidDataset
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader
from utils import nms, mean_average_precision

# Initialize Dataset
train_dataset = TrainDataset('dataset/train_img', 'dataset/train_label')
valid_dataset = ValidDataset('dataset/valid_img', 'dataset/valid_label')
# Build DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))
train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
valid_data_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

# Check GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''images, targets = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
img = images[0].permute(1,2,0).cpu().numpy()
fix, ax = plt.subplots(1, 1, figsize=(12, 6))
for box in boxes:
    cv2.rectangle(img,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220/255, 0, 0), 1)
ax.set_axis_off()
ax.imshow(img)
plt.show()'''

# Establish model pretrained with Imagenet
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
#model.roi_heads.box_predictor.cls_score.weight.data.fill_(0)
#model.roi_heads.box_predictor.cls_score.bias.data.fill_(0.01)
model = model.to(device)

# Hyperparameters
num_epochs = 10
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

# Train function
def train(epoch, trainloader, model, optimizer):
    model.train()
    itr = 1
    avg_loss = 0
    for images, targets in tqdm(trainloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print(len(images))
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        avg_loss += loss_value

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} train_loss: {avg_loss / itr}")

        itr += 1

    print(f"Epoch #{epoch} train_loss: {avg_loss / itr}")
    return avg_loss / itr

# Valid function
def valid(epoch, validationloader, model):
    model.eval()
    itr = 1
    avg_mAP = 0
    with torch.no_grad():
        for images, targets in tqdm(validationloader):

            images = list(image.to(device) for image in images)
            output = model(images)
            boxess = []
            for idx in range(4):
                boxes = output[idx]['boxes'].data.cpu().numpy()
                scores = output[idx]['scores'].data.cpu().numpy()
                boxess.append(nms(idx, boxes, scores, iou_threshold=0.5, threshold=0.4))
            avg_mAP += mean_average_precision(list(boxess), list(targets), iou_threshold=0.5, num_classes=2)
            if itr % 10 == 0:
                sample = images[3].permute(1, 2, 0).cpu().numpy()
                fog, ax = plt.subplots(1, 1, figsize=(12, 6))
                for box in boxes:
                    cv2.rectangle(sample,
                                  (box[0], box[1]),
                                  (box[2], box[3]),
                                  (220 / 255, 0, 0), 2)
                ax.set_axis_off()
                ax.imshow(sample)
                print(avg_mAP / itr)
                plt.savefig('./output/label'+str(epoch)+str(itr)+'.jpg')
                plt.close('all')
            if itr > 100:
                break
            itr += 1
    print(f"Epoch #{epoch} valid_mAP: {avg_mAP / itr}")
    return avg_mAP / itr

# Start training
train_loss = []
valid_mAP = []
for epoch in range(num_epochs):

    tra_loss = train(epoch, train_data_loader, model, optimizer)
    val_loss = valid(epoch, valid_data_loader, model)
    train_loss.append(tra_loss)
    valid_mAP.append(val_loss)

    lr_scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])

plt.plot(train_loss)
plt.show()
plt.plot(valid_mAP)
plt.show()

# Savemodel
torch.save(model.state_dict(), 'model.pth')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'ckpt.pth')


