import os
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from natsort import natsorted
from torch.utils.data import Dataset


def gaussian(num_theta, num_rho, center, sig):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    # create nxn zeros
    y = np.linspace(0, num_theta - 1, num_theta)
    x = np.linspace(0, num_rho - 1, num_rho)
    x, y = np.meshgrid(x, y)
    x0 = center[1]
    y0 = center[0]
    res = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sig**2))

    return res


class BaseDataset(Dataset):
    def __init__(
        self,
        split,
        size,  # H, W
        num_angle=180,
        num_rho=100,
        augment=False,
    ):
        # if split not in ["train", "val", "test", "challenge", "normal", "a", "A"]:
        #     raise ValueError("Please check the split.")

        super(BaseDataset, self).__init__()
        self.num_angle = num_angle
        self.num_rho = num_rho
        self.size = size  # H, W
        self.resize = (size[1], size[0])  # W, H
        self.augment = augment

    def calc_coords(self, label):
        """
        calulate the coordinates of the beginning and the end of the needle

        Args:
            label (_type_): _description_

        Returns:
            y0, x0, y1, x1: location of points in the image space.
                The origin is the midpoint of the image. The x axis is from left to the right, the y axis is from the top to the bottom.
        """
        H, W = self.size
        coords = np.argwhere(label)
        try:
            x0 = coords[:, 1].min()
            x1 = coords[:, 1].max()
            y0 = coords[coords[:, 1] == x0][:, 0].min()
            y1 = coords[coords[:, 1] == x1][:, 0].max()

            x0 -= W / 2
            x1 -= W / 2
            y0 -= H / 2
            y1 -= H / 2
        except ValueError:
            x0, y0, x1, y1 = 0, 0, 0, 0

        return x0, y0, x1, y1

    def calc_rho_theta(self, x0, y0, x1, y1):
        """
        calculate the rho and theta of the line

        Args:
            y0 (_type_): _description_
            x0 (_type_): _description_
            y1 (_type_): _description_
            x1 (_type_): _description_

        Returns:
            theta, rho: _description_
        """
        # hough transform
        theta = np.arctan2(y1 - y0, x1 - x0) + np.pi / 2
        rho = x0 * np.cos(theta) + y0 * np.sin(theta)
        return theta, rho

    def line_shaft(self, theta, rho):
        """
        create the hough space label for the shaft

        Args:
            theta (_type_): _description_
            rho (_type_): _description_

        Returns:
            hough_space_shaft, theta, rho:
                - hough_space_shaft: the hough space label for the shaft, which is a gaussian distribution
                - theta: the index of theta in the hough space
                - rho: the index of rho in the hough space
        """
        # rho is the distance from the line to the middle point of the image
        H, W = self.size
        # calculate resolution of rho and theta
        irho = np.sqrt(H * H + W * W) / self.num_rho
        itheta = np.pi / self.num_angle

        # rho can be a negative value, so we need to shift the index
        rho_idx = int(np.round(rho / irho)) + int((self.num_rho) / 2)
        theta_idx = int(np.round(theta / itheta))
        if theta_idx >= self.num_angle:
            theta_idx = self.num_angle - 1
        hough_space_shaft = gaussian(self.num_angle, self.num_rho, (theta_idx, rho_idx), sig=2)

        return hough_space_shaft, theta_idx, rho_idx

    def all_line_cross_tip(self, y, x):
        """
        create the hough space label for the tip. The tip is the intersection of all the lines.

        Args:
            y (_type_): _description_
            x (_type_): _description_

        Returns:
            hough_space_tip: each row is the hough space label for each line, which is a gaussian distribution
        """
        H, W = self.size
        irho = np.sqrt(H * H + W * W) / self.num_rho

        hough_space_tip = np.zeros((self.num_angle, self.num_rho))
        for i in range(self.num_angle):
            theta = i * np.pi / self.num_angle
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho = int(np.round(rho / irho)) + int((self.num_rho) / 2)
            hough_space_tip[i] = gaussian(1, self.num_rho, (0, rho), sig=3)
        return hough_space_tip

    def process_label(self, label):
        """
        This function is the real entry of the dataset.
        It will process the label image to get the hough space label and the theta and rho of the line and the tip location.

        Args:
            label (_type_): _description_

        Returns:
            hough_space_label, theta, rho, tip: _description_
        """
        # find the coordinates of the line
        x0, y0, x1, y1 = self.calc_coords(label)
        # H, W = self.size
        # cv2.line(img, (int(x0 + W / 2), int(y0 + H / 2)), (int(x1 + W / 2), int(y1 + H / 2)), 255, 2)
        # cv2.imwrite('coordscheck.jpg',img)

        # no line in the image
        if y0 == y1 and x0 == x1:
            return np.zeros((2, self.num_angle, self.num_rho)), 0, 0

        # calculate the rho and theta
        # rho is the distance from the line to the middle of the image
        theta, rho = self.calc_rho_theta(x0, y0, x1, y1)
        # cos = np.cos(theta)
        # sin = np.sin(theta)
        # x0 = cos * rho
        # y0 = sin * rho
        # x1 = int(x0 + 1000 * (-sin))
        # y1 = int(y0 + 1000 * cos)
        # x2 = int(x0 - 1000 * (-sin))
        # y2 = int(y0 - 1000 * cos)
        # cv2.line(img, (int(x1 + W / 2), int(y1 + H / 2)), (int(x2 + W / 2), int(y2 + H / 2)), 255, 2)
        # cv2.imwrite("houghlinescheck.jpg", img)

        # create the hough space label
        hough_space_label = np.zeros((2, self.num_angle, self.num_rho))
        hough_space_label[0], theta, rho = self.line_shaft(theta, rho)

        # sort (y0, x0) and (y1, x1) based on y
        if y0 > y1:
            hough_space_label[1] = self.all_line_cross_tip(y0, x0)
            tip = np.array([y0, x0])
        else:
            hough_space_label[1] = self.all_line_cross_tip(y1, x1)
            tip = np.array([y1, x1])

        # y0, x0, y1, x1 was calculated by seen the middle point of the image as the origin
        # tip location in the tensor space
        tip[0] += self.size[0] / 2
        tip[1] += self.size[1] / 2

        return hough_space_label, theta, rho, tip


class ImgDataset(BaseDataset):
    """
    Use this dataset if the input of the model requires single image + single label.
    The dataset directory should be organized as follows:
    -- data_path
     |-- imgs
     |-- annos
    The file names of the images and their labels should be the same.

    Args:
        data_path (_type_): Path to the dataset
        split (_type_): Aplit of the dataset, should be either 'train', 'val' or 'test'
        size (tuple, optional): The size of the output image. If the size is different from the original image, the image will be resized.
        num_angle (int, optional): Number of angles in the prediction. Defaults to 180.
        num_rho (int, optional): Number of rhos in the prediction. Defaults to 100.
        augment (bool, optional): Augmentation is required or not. Defaults to False.
            Augmentation includes:
            - horizontally flip of the images
    """

    def __init__(
        self,
        data_path,
        split,
        size,  # H, W
        num_angle=180,
        num_rho=100,
        augment=False,
    ):
        super(ImgDataset, self).__init__(split, size, num_angle, num_rho, augment)
        self.data_path = Path(data_path)
        self.img_path = self.data_path / "imgs"
        self.annos_path = self.data_path / "annos"

        self.seq_names = natsorted(
            open(Path(data_path) / f"{split}.txt").read().splitlines()
        )
        self.all_file_names = [
            natsorted(os.listdir(self.img_path / name)) for name in self.seq_names
        ]
        self.length_list = [
            len(os.listdir(self.img_path / name)) for name in self.seq_names
        ]

    def __len__(self):
        return sum(self.length_list)

    def __getitem__(self, index):
        # TODO: rewrite this function
        i = 0
        while self.length_list[i] <= index:
            index -= self.length_list[i]
            i += 1
        file_name = self.all_file_names[i][index]

        img = cv2.imread(
            str(self.img_path / self.seq_names[i] / file_name), cv2.IMREAD_GRAYSCALE
        )
        print(img.shape)
        label = cv2.imread(
            str(self.annos_path / (self.seq_names[i] + ".png")), cv2.IMREAD_GRAYSCALE
        )

        img = cv2.resize(img, self.resize)
        label = cv2.resize(label, self.resize)

        if self.augment:
            img, label = self.aug(img, label)

        hough_space_label, theta, rho, tip = self.process_label(label)

        return (
            np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1.0,
            hough_space_label,
            label,
            theta,
            rho,
            tip,
        )

    def aug(self, img, label):
        augseq = iaa.Sequential(
            [
                # iaa.CropAndPad(percent=(-0.25, 0.25)),  # crop and pad images
                # iaa.Affine(rotate=(-25, 25)),  # rotate the image
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                # iaa.Multiply((0.8, 1.2)),  # change brightness of images
            ]
        )
        label = SegmentationMapsOnImage(label, shape=label.shape)
        img, label = augseq(image=img, segmentation_maps=label)
        label = label.get_arr()
        # visualize the augmented label
        # cv2.imwrite("augcheck.jpg", label * 255)

        return img, label


class SeqDataset(BaseDataset):
    """
    Use this dataset if the input of the model requires image sequence + label of the last image in the sequence.
    AND each video shared the same label.
    If you want to use two datasets together, you can use torch.utils.data.ConcatDataset.
    The dataset directory should be organized as follows:
    -- data_path
        |-- imgs
        |-- seq_1: images of a video
        |-- seq_2: images of another video
        |-- ...
        |-- annos: labels of all videos (seq_1.png, seq_2.png, ...)
    The file names of the sequnces and their label images should be the same.

    Args:
        data_path (_type_): Path to the dataset
        split (_type_): Split of the dataset, should be either 'train', 'val' or 'test'
        size (tuple, optional): The size of the output image. If the size is different from the original image, the image will be resized.
        num_angle (int, optional): Number of angles in the prediction. Defaults to 180.
        num_rho (int, optional): Number of rhos in the prediction. Defaults to 100.
        augment (bool, optional): Augmentation is required or not. Defaults to True.
            Augmentation includes:
            - horizontally flip of the images
    """

    def __init__(
        self,
        data_path,
        split,
        size,  # H, W
        mask_size,
        seq_length=30,
        num_angle=180,
        num_rho=100,
        augment=True,
    ):
        super().__init__(split, size, num_angle, num_rho, augment)

        self.resize_mask = (mask_size[1], mask_size[0])
        self.data_path = Path(data_path)
        self.seq_length = seq_length

        self.img_path = self.data_path / "pork"
        self.anno_path = self.data_path / "gt"

        self.seq_names = natsorted(
            [line.strip() for line in open(Path(data_path) / f"{split}.txt").read().splitlines() if line.strip()]
        )
        self.all_file_names = [
            natsorted(os.listdir(self.img_path / name)) for name in self.seq_names
        ]

        self.length_list = [
            len(os.listdir(self.img_path / name)) - 35 + 1
            for name in self.seq_names
        ]

    def aug(self, img_seq, label):
        augseq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.LinearContrast((0.8, 1.2)),  # improve or worsen the contrast
                iaa.Multiply((0.8, 1.2)),  # change brightness
                # iaa.GaussianBlur(
                #     sigma=(0, 2.0)
                # ),  # blur images with a sigma of 0 to 3.0
            ]
        )
        # fix the random state for all the augmentations
        augseq = augseq.to_deterministic()

        img_seq_aug = []
        label_aug = []

        for i in range(len(img_seq)):
            img = augseq(image=img_seq[i])
            img_seq_aug.append(img)

        for i, l in enumerate(label):  # label is a list of two masks: mask1 and mask2
            label_frame = SegmentationMapsOnImage(l, shape=l.shape)  # Convert to imgaug format
        # Apply augmentation to the respective frames: mask1 -> 30th frame, mask2 -> 35th frame
            if i == 0:  # Apply to mask1 (30th frame)
                img, label_frame = augseq(images=img_seq[29], segmentation_maps=label_frame)
            elif i == 1:  # Apply to mask2 (35th frame)
                img, label_frame = augseq(images=img_seq[34], segmentation_maps=label_frame)
        
            label_aug.append(label_frame.get_arr())  # Collect the augmented labels
        return img_seq_aug, np.array(label_aug)
        # label = SegmentationMapsOnImage(label, shape=label.shape)
        # img, label = augseq(images=img_seq[-1], segmentation_maps=label)
        # img_seq_aug.append(img)
        # label = label.get_arr()
        # # visualize the augmented label
        # # cv2.imwrite("augcheck_img.jpg", img_seq_aug[-1])
        # # cv2.imwrite("augcheck_label.jpg", label * 255)

        # return img_seq_aug, label

    def __len__(self):
        return sum(self.length_list)

    def __getitem__(self, index):
        i = 0
        while self.length_list[i] <= index:
            index -= self.length_list[i]
            i += 1

        seq_file_names = self.all_file_names[i][index : index + 35]
        assert len(seq_file_names) == 35, "sequence length not match"

        img_seq = []
        for file_name in seq_file_names:
            if file_name is None:
                raise ValueError(f"Invalid file name '{file_name}' in sequence: {seq_file_names}")
            image_path = str(self.img_path / self.seq_names[i] / file_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Failed to load image: {image_path}")
            img = cv2.resize(img, self.resize)
            img_seq.append(img)

        label1_path = str(self.anno_path / self.seq_names[i] / seq_file_names[29]) 
        label2_path = str(self.anno_path / self.seq_names[i] / seq_file_names[34]) 
        
        mask1 = cv2.imread(label1_path, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(label2_path, cv2.IMREAD_GRAYSCALE)

        mask1 = cv2.resize(mask1, self.resize_mask)
        mask2 = cv2.resize(mask2, self.resize_mask)
        
        if self.augment:
            # img_seq, label = self.aug(img_seq, label)
            img_seq, [mask1, mask2] = self.aug(img_seq, [mask1, mask2])

        img_seq = np.expand_dims(np.array(img_seq), 1).astype(np.float32) / 127.5 - 1.0
        # mask1 = np.expand_dims(mask1, 0).astype(np.float32) / 255.0
        # mask2 = np.expand_dims(mask2, 0).astype(np.float32) / 255.0

        return img_seq[:30], img_seq[5:35], np.stack([mask1, mask2], axis=0)


if __name__ == "__main__":
    # can used for test dataset and utils
    import torch
    from torch.utils.data import DataLoader

    from utils import reverse_all_hough_space, reverse_max_hough_space, vis_result

    img_dataset = ImgDataset(
        data_path="dataset/Beef",
        split="test",
        size=(657 // 2, 671 // 2),
        num_angle=180,
        num_rho=100,
        augment=True,
    )
    img_dataloader = DataLoader(img_dataset, batch_size=1, shuffle=True)
    for batch in img_dataloader:
        img, hough_space_label, label, theta, rho, tip = batch
        img = img[0][0]
        # print(img.shape)
        hough_space_label_shaft = hough_space_label[0][0]
        hough_space_label_tip = hough_space_label[0][1]
        label = label[0]
        theta = theta[0]
        rho = rho[0]
        tip = tip[0]
        lines = reverse_all_hough_space(
            torch.zeros(img.shape[-2:]), hough_space_label_tip, 180, 100
        )
        img_tip = vis_result(img, lines, label)
        W, H = img.shape[-2:]
        # find the max value's location in the lines, using pytorch
        tip_loc = torch.argmax(lines)
        x_pos = tip_loc / H
        y_pos = tip_loc % H
        # print(x_pos, y_pos)
        # print(tip)
        cv2.imwrite("img_tip.png", img_tip)
        line = reverse_all_hough_space(
            torch.zeros(img.shape[-2:]), hough_space_label_shaft, 180, 100
        )
        img_shaft = vis_result(img, line, label)
        cv2.imwrite("img_shaft.png", img_shaft)
        break

    seq_dataset = SeqDataset(
        data_path="dataset/Beef",
        split="test",
        size=(657 // 2, 671 // 2),
        seq_length=30,
        num_angle=180,
        num_rho=100,
        augment=True,
    )
    # print(len(seq_dataset))
    seq_dataloader = DataLoader(seq_dataset, batch_size=2, shuffle=True)
    for batch in seq_dataloader:
        img, hough_space_label, label, theta, rho, tip = batch
        img = img[0][-1]
        # print(img.shape)
        hough_space_label_shaft = hough_space_label[0][0]
        hough_space_label_tip = hough_space_label[0][1]
        label = label[0]
        theta = theta[0]
        rho = rho[0]
        tip = tip[0]

        lines = reverse_all_hough_space(
            torch.zeros(img.shape[-2:]), hough_space_label_tip, 180, 100
        )
        seq_tip = vis_result(img, lines, label)
        W, H = img.shape[-2:]
        # find the max value's location in the lines, using pytorch
        tip_loc = torch.argmax(lines)
        x_pos = tip_loc / H
        y_pos = tip_loc % H
        # print(x_pos, y_pos)
        # print(tip)
        cv2.imwrite("seq_tip.png", seq_tip)
        line = reverse_max_hough_space(
            torch.zeros(img.shape[-2:]), hough_space_label_shaft, 180, 100
        )
        seq_shaft = vis_result(img, line, label)
        cv2.imwrite("seq_shaft.png", seq_shaft)
        break
