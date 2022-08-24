from __future__ import annotations
import os
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import transforms as T
import utils

import xml.etree.ElementTree as ET

from engine import train_one_epoch, evaluate


class FaceMaskDataset(torch.utils.data.Dataset):
    """Dataloader for extract, transform, and load the data in the model"""

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])

        img = Image.open(img_path).convert("RGB")
        annotation = ET.parse(annotation_path).getroot()

        # obj_ids = np.unique(mask)

        # obj_ids = obj_ids[1:]

        # masks = mask == obj_ids[:, None, None]
        names = {"with_mask": 1, "without_mask": 2, "mask_weared_incorrect": 3}
        num_objs = len(annotation.findall("object"))

        labels = []
        boxes = []
        for child in annotation.findall("object"):
            labels.append(names[child.find("name").text])
            xmin = int(child.find("bndbox").find("xmin").text) - 1
            ymin = int(child.find("bndbox").find("ymin").text) - 1
            xmax = int(child.find("bndbox").find("xmax").text) - 1
            ymax = int(child.find("bndbox").find("ymax").text) - 1
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # labels = torch.ones(num_objs, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}

        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def load_backbone():
    "Loads a backbone from a torchvision model"
    # backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone = torchvision.models.mobilenet_v2(
        weights="DEFAULT"
    ).features  # Refinement
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=4,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model


def get_transform(train):
    "Returns a callable transform to be used for preprocessing images."
    transforms = []
    transforms.append(T.ToTensor())
    # transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def calculate_metrics(model, test, device = "cuda"):
    "Calculates the metrics for the model"

    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for data, target in test:
            prediction = model([data.to(device)])
            labels_quantity = len(target["labels"])
            total += labels_quantity
            corrects += (
                prediction[0]["labels"].cpu()[:labels_quantity]
                .eq(target["labels"].cpu())
                .sum()
                .item()
            )
    accuracy = corrects / total * 100
    print("Accuracy of the model: {}".format(accuracy))
    return accuracy


if __name__ == "__main__":

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # our dataset has two classes only - background and person
    num_classes = 4
    # use our dataset and defined transformations
    dataset = FaceMaskDataset(".", get_transform(train=True))
    dataset_test = FaceMaskDataset(".", get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # get the model using our helper function
    model = load_backbone()

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0001)  # Refinement
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    
    # calculate metrics
    torch.save(model, "models/fasterrcnn_model.pth")
    
    calculate_metrics(model, dataset_test)

    print("That's it!")
