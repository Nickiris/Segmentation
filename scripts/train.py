import os
import torch
import torchvision
from models.fcn import FCN8s
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.pascal_voc import collate_fn, makecmap, VOCSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)[0]
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch += 1
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)[0]
            test_loss += loss_fn(pred, y).item()
    #         correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")



def main():
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'potted plant',
                 'sheep', 'sofa', 'train', 'monitor']
    color_maps = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
     [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
     [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
     [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
     [0, 192, 0], [128, 192, 0], [0, 64, 128]]


    base_path = r"C:\Users\12112\iPython\Jupyter\Deep Learning\data\Pascal VOC 2012\train"
    img_path = os.path.join(base_path, "JPEGImages")
    mask_path = os.path.join(base_path, "SegmentationClass")
    train_sets = os.path.join(base_path, "ImageSets/Segmentation/train.txt")
    val_sets = os.path.join(base_path, "ImageSets/Segmentation/val.txt")

    with open(train_sets,'r') as f:
        train_sets = f.read().split('\n')
    with open(val_sets,'r') as f:
        val_sets = f.read().split('\n')
    backbone = torchvision.models.vgg.vgg16(pretrained=True).features
    model = FCN8s(num_classes=len(classes), bakcbone=backbone)

    cm2lbl = makecmap(color_maps,size=(3, 256, 256))

    transform = transforms.Compose([transforms.ToTensor(),# PIL or ndarray
                                    transforms.CenterCrop((256, 256)),  # tensor
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    train_data = VOCSegmentation(img_path, mask_path, train_sets, cm2lbl,transform)
    val_data = VOCSegmentation(img_path, mask_path, val_sets, cm2lbl,transform)

    train_dataloader = DataLoader(train_data,
                             shuffle=True,
                             batch_size=4)
    val_dataloader = DataLoader(val_data,
                                shuffle=True,
                                batch_size=4)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    epochs = 2
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(val_dataloader, model, loss_fn)
    print("Done!")


if __name__ == '__main__':
    main()