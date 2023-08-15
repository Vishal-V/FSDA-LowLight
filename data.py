import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import cv2
import rawpy
import random
from PIL import Image


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )

    # print (np.max(out))
    return out


def pack_nikon(raw, resize=False):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 600, 0) / (16383 - 600)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    if resize:
        out = cv2.resize(out, (out.shape[1] // 4, out.shape[0] // 4))

    return out


def pack_canon(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 2048, 0) / (16383 - 2048)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    return out


class SonyDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.ARW")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Sony images onto RAM....")
        for i in range(len(self.ids)):
            # input
            index = self.ids[i]
            print(index, i)
            in_files = glob.glob(self.input_dir + "%05d_00*.ARW" % index)
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
            in_fn = os.path.basename(in_path)
            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.ARW" % index)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(f"Loaded all {len(self.ids)} Sony images onto RAM....")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        index = self.ids[ind]

        # crop
        ratio = self.ratios[ind]
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, ind)
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, index, ratio


class NikonTrainSet(Dataset):
    """Training as target domain with Fuji as source"""

    def __init__(
        self,
        input_dir,
        gt_dir,
        no_of_items=1,
        set_num=1,
        ps=512,
        random_sample=False,
        seed=0,
        stratify=False,
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.NEF")
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [4]
            elif set_num == 2:
                self.ids = [29]
            elif set_num == 3:
                self.ids = [14]

        elif no_of_items == 2:
            if set_num == 1:
                self.ids = [4, 15]
            elif set_num == 2:
                self.ids = [24, 29]
            elif set_num == 3:
                self.ids = [22, 33]

        elif no_of_items == 4:
            if set_num == 1:
                self.ids = [4, 15, 24, 29]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54]

        elif no_of_items == 6:
            if set_num == 1:
                self.ids = [4, 15, 24, 29, 33, 57]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57, 24, 13]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54, 25, 30]

        elif no_of_items == 8:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and not random_sample:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49, 15, 14]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49, 33, 57]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26, 33, 13]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )

        elif no_of_items == 16 and random_sample == False:
            if set_num == 1:
                self.ids = [
                    4,
                    13,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    39,
                    19,
                    59,
                    17,
                ]
            elif set_num == 2:
                self.ids = [
                    25,
                    30,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    38,
                    44,
                    6,
                    16,
                ]
            elif set_num == 3:
                self.ids = [4, 13, 22, 24, 39, 15, 29, 46, 54, 17, 7, 26, 38, 44, 6, 16]

        elif no_of_items == 16 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

        print("Loading Nikon images onto RAM....")
        for i in range(len(self.ids)):

            index = self.ids[i]
            in_files = glob.glob(self.input_dir + "%05d_0*.NEF" % id)
            in_files = sorted(in_files)

            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            gt_files = glob.glob(self.gt_dir + "%05d_0*.NEF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            print(set_num, index, i, ratio)

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_nikon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Nikon images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        ind = ind % 4
        index = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, index, self.ratios[ind]


class CanonTrainSet(Dataset):
    """Training as target domain with Fuji as source"""

    def __init__(self, input_dir, gt_dir, no_of_items=1, set_num=1, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "1*.CR2")  # file names
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [10513]
            elif set_num == 2:
                self.ids = [10510]
            elif set_num == 3:
                self.ids = [10518]

        elif no_of_items == 3:
            if set_num == 1:
                self.ids = [10513, 10519, 10502]
            elif set_num == 2:
                self.ids = [10506, 10510, 10520]
            elif set_num == 3:
                self.ids = [10514, 10507, 10518]

        elif no_of_items == 6:
            if set_num == 1:
                self.ids = [10513, 10519, 10502, 10506, 10510, 10520]
            elif set_num == 2:
                self.ids = [10516, 10518, 10504, 10507, 10514, 10517]
            elif set_num == 3:
                self.ids = [10503, 10524, 10501, 10500, 10522, 10523]

        elif no_of_items == 9:
            if set_num == 1:
                self.ids = [
                    10507,
                    10520,
                    10510,
                    10502,
                    10506,
                    10519,
                    10513,
                    10514,
                    10517,
                ]
            elif set_num == 2:
                self.ids = [
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                    10516,
                    10518,
                    10504,
                ]
            elif set_num == 3:
                self.ids = [
                    10516,
                    10518,
                    10504,
                    10507,
                    10514,
                    10517,
                    10513,
                    10519,
                    10502,
                ]

        elif no_of_items == 12:
            if set_num == 1:
                self.ids = [
                    10516,
                    10518,
                    10504,
                    10507,
                    10514,
                    10517,
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                ]
            elif set_num == 2:
                self.ids = [
                    10516,
                    10523,
                    10507,
                    10524,
                    10501,
                    10503,
                    10500,
                    10517,
                    10504,
                    10518,
                    10514,
                    10522,
                ]
            elif set_num == 3:
                self.ids = [
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                    10503,
                    10524,
                    10501,
                    10500,
                    10522,
                    10523,
                ]

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.input_images = [None] * len(self.ids)
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Canon images into RAM....")
        for i in range(len(self.ids)):
            index = self.ids[i]
            print(index, i)

            # Input
            in_files = glob.glob(self.input_dir + "%05d_00*.CR2" % id)
            in_files = sorted(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            # Ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.CR2" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # Ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_canon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Canon images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        index = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # Random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # Random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, index, self.ratios[ind]

