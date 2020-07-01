import random
import numpy as np
import json
import os
import cv2
import albumentations as A

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class jsonDataset(data.Dataset):
    def __init__(self, path, classes, transform, input_image_size, do_aug=False, view_image=False):
        self.path = path
        self.classes = classes
        self.transform = transform
        self.input_size = input_image_size
        self.view_img = view_image
        self.do_aug = do_aug

        self.fnames = list()
        self.labels = list()

        self.num_classes = len(self.classes)

        fp_read = open(self.path, 'r')
        gt_dict = json.load(fp_read)

        all_labels = list()
        all_img_path = list()

        # read gt files
        for gt_key in gt_dict:
            gt_data = gt_dict[gt_key][0]

            # img = cv2.imread(gt_data['image_path'])
            # img_rows = img.shape[0]
            # img_cols = img.shape[1]

            class_name = gt_data['label']
            if class_name not in self.classes:
                print('weired class name: ' + class_name)
                print(gt_data['image_path'])
                continue

            class_idx = self.label_map(class_name)
            all_labels.append(class_idx)
            all_img_path.append(gt_data['image_path'])

        if len(all_labels) == len(all_img_path):
            num_images = len(all_img_path)
        else:
            print('num. of labels: ' + str(len(all_labels)))
            print('num. of paths: ' + str(len(all_img_path)))
            raise ValueError('num. of elements are different(all boxes, all_labels, all_img_path)')

        for idx in range(0, num_images, 1):
            self.fnames.append(all_img_path[idx])
            self.labels.append(torch.tensor(all_labels[idx], dtype=torch.int64))

        self.num_samples = len(self.fnames)

        if self.do_aug is True:
            bbox_params = A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.3)
            self.augmentation = A.Compose([
                # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                # A.MotionBlur(blur_limit=11, p=0.5),
                # A.GaussNoise(var_limit=130, p=0.5),

                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
                A.ChannelShuffle(p=1.0),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.15, 0.15), rotate_limit=45, p=0.5,
                                   border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.5),

                # A.Normalize(mean=(0.407, 0.458, 0.482), std=(1.0, 1.0, 1.0)),
            ], bbox_params=bbox_params, p=1.0)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        labels = self.labels[idx]
        img = cv2.imread(fname)

        if self.do_aug is True:
            bboxes = [bbox.tolist() + [label.item()] for bbox, label in zip(boxes, labels)]
            augmented = self.augmentation(image=img, bboxes=bboxes)
            img = np.ascontiguousarray(augmented['image'])
            boxes = augmented['bboxes']
            boxes = [list(bbox) for bbox in boxes]
            labels = [bbox.pop() for bbox in boxes]
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        img = self.transform(img)

        return img, labels, fname

    def __len__(self):
        return self.num_samples


    def collate_fn(self, batch):

        return inputs, torch.stack(loc_targets), torch.stack(cls_targets), mask_targets, paths


def test():
    import torchvision

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    # ])
    # set random seed
    random.seed(3000)
    np.random.seed(3000)
    torch.manual_seed(3000)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    classes = 'aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor'
    classes = classes.split('|')

    dataset = jsonDataset(path='data/voc07_trainval.json', classes=classes, transform=transform,
                          input_image_size=(328, 328), num_crops=-1, view_image=True, do_aug=True)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0,
                                             collate_fn=dataset.collate_fn)

    while True:
        for idx, (images, loc_targets, cls_targets, mask_targets, paths) in enumerate(dataloader):
            np_img = images.numpy()
            print(images.size())
            print(loc_targets.size())
            print(cls_targets.size())
            print(mask_targets.size())
            # break

if __name__ == '__main__':
    test()
