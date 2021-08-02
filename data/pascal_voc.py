import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    # could multi-scale train
    return tuple(zip(*batch))

def makecmap(color_maps, size):
    cm2lbl = torch.zeros(size[-1]**3, dtype=torch.long)
    for i, cm in enumerate(color_maps):
        cm2lbl[(cm[0] * size[-2] + cm[1]) * size[-1] + cm[2]] = i  # 建立索引

    return cm2lbl

def mask2label(mask, cm2lbl):
    size = mask.size()[-2:]
    idx = (mask[0,:,:] * size[-2] + mask[1,:,:]) * size[-1] + mask[2, :, :]
    labels = cm2lbl[idx.long()]
    return labels


class VOCSegmentation(Dataset):
    def __init__(self,  img_path, mask_path, img_sets, cm2lbl, transforms=None):
        self.img_paths = img_path
        self.mask_paths = mask_path
        # list
        self.img_sets = img_sets
        self.cm2lblb = cm2lbl
        self.transforms = transforms

    def __getitem__(self, idx):
        image = cv2.imread(f'{self.img_paths}/{self.img_sets[idx]}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = None
        if self.mask_paths is not None:
            mask = cv2.imread(f'{self.mask_paths}/{self.img_sets[idx]}.png', cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = torch.as_tensor(mask)

        label = None
        if self.transforms:
            image = self.transforms(image)
            size = image.size()[-2:]
            resize = transforms.CenterCrop(size)
            if mask is not None:
                mask = mask.permute(2, 0, 1)
                mask = resize(mask)
                label = mask2label(mask, self.cm2lblb)

        return image, label

    def __len__(self):
        return len(self.img_sets)

if __name__ == "__main__":
    base_path = r"C:\Users\12112\iPython\Jupyter\Deep Learning\data\Pascal VOC 2012\train"
    img_path = os.path.join(base_path, "JPEGImages")
    mask_path = os.path.join(base_path, "SegmentationClass")
    img_sets = os.path.join(base_path, "ImageSets/Segmentation/val.txt")
    with open(img_sets,'r') as f:
        img_sets = f.read().split('\n')
    # ToTensor() convert shape(H, W, C) array to shape(C, H, W) tensor,and divided 255.
    transform = transforms.Compose([transforms.ToTensor(),# PIL or ndarray
                                    transforms.CenterCrop((256, 256)),  # tensor
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
    classes__ = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'potted plant',
                 'sheep', 'sofa', 'train', 'monitor']
    color_maps = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
     [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
     [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
     [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
     [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    cm2lbl = makecmap(color_maps, (3,256, 256))
    dataset = VOCSegmentation(img_path, mask_path, img_sets, cm2lbl,transform)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=4)
    imgs = []
    labels = []
    for i, data in enumerate(dataloader):
        print(data[0].shape)
        print(data[1].shape)
        break
    out = torch.rand(4, 5, 256, 256)
    # num_classes的大小不会小于label中的最大值
    print(F.cross_entropy(out, label))




