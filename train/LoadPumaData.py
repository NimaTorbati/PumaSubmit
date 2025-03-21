import os
from roifile import ImagejRoi
from matplotlib import pyplot as plt
import cv2
import numpy as np
import geojson
from skimage.draw import polygon2mask
import torch
import cv2
from torch.utils.data import Dataset
from random_augs_for_hist import distort_image_with_randaugment
from torchvision.transforms import InterpolationMode
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import imgaug as ia
from imgaug import augmenters as iaa
from augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,

)

import kornia
import kornia.geometry.transform as T


def load_data_tissue(target_size = (128,128), data_path = '', annot_path = '',tissue_labels = None,im_size = (1024,1024)):

    mask = np.zeros((target_size[0], target_size[1],len(tissue_labels)))
    y_all = np.zeros((len(data_path),target_size[0], target_size[1],len(tissue_labels)))

    if target_size[0] == im_size[0]:
        X_all = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) for image in
                   data_path]
    else:
        X_all = [cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), target_size) for image in
                   data_path]




    for i in range(len(annot_path)):
        print(i)
        annotation_geojson_file = open(annot_path[i])
        collection = geojson.load(annotation_geojson_file)
        try:
            if "features" in collection.keys():
                collection = collection["features"]
            elif "geometries" in collection.keys():
                collection = collection["geometries"]
        except AttributeError:
            # already a list?
            pass
        for j in range(len(collection)):
            for k in range(len(collection[j].geometry.coordinates)):
                try:
                    polygon_numpy = np.array(collection[j].geometry.coordinates[k])
                except:
                    polygon_numpy = np.array(collection[j].geometry.coordinates[k][0])
                polygon_numpy = np.squeeze(polygon_numpy)
                polygon_numpy1 = polygon_numpy.copy()
                polygon_numpy1[:, 0] = polygon_numpy[:, 1]
                polygon_numpy1[:, 1] = polygon_numpy[:, 0]
                polygon_numpy1 = np.squeeze(polygon_numpy1)
                mask_sample = polygon2mask((im_size[0], im_size[1]), polygon_numpy1)
                ind = tissue_labels.index(collection[j].properties['classification']['name'])
                mask[:,:,ind] = ((cv2.resize(np.array(mask_sample,dtype=np.float32), target_size) - np.sum(mask,axis=2))>0).astype('float32')*ind + mask[:,:,ind]

        y_all[i,:,:,:] = mask
        mask = 0*mask


    X_all = np.array(X_all)
    X_all = X_all.astype('float32')
    y_all = y_all.astype('int64')
    return X_all, y_all


def piecewise_affine_transform(image: torch.Tensor, mask: torch.Tensor, num_points: int = 10) -> tuple:
    """
    Apply piecewise affine transform using Thin-Plate Splines on the image and mask.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W) with dtype float32.
        mask (torch.Tensor): Input mask tensor of shape (B, 1, H, W) with dtype torch.long.
        num_points (int): Number of control points for TPS.

    Returns:
        transformed_image (torch.Tensor): Transformed image tensor.
        transformed_mask (torch.Tensor): Transformed mask tensor.
    """
    B, C, H, W = image.shape

    # Generate random control points in normalized coordinates [-1, 1]
    src_points = torch.rand((B, num_points, 2), device=image.device) * 2 - 1  # Random points
    dst_points = src_points + (torch.randn_like(src_points) * 0.02)  # Slight perturbation

    # Apply TPS transformation
    # Create TPS transformation for images
    tps_image = T.warp_image_tps(image, src_points, dst_points, normalized_coordinates=True)

    # Create TPS transformation for masks
    # Convert mask to float32 for transformation and back to long after
    mask_float = mask.float()
    tps_mask = T.warp_image_tps(mask_float, src_points, dst_points, normalized_coordinates=True)
    tps_mask = torch.round(tps_mask).long()  # Convert back to long for masks

    return tps_image, tps_mask








def augs_mine(images = None, masks = None, num_classes = 6):
    i_m = np.zeros((np.shape(images)[0], np.shape(images)[1], np.shape(images)[2] + 1))
    ag1 = iaa.PiecewiseAffine(scale=(0,0.1), seed=0, order=0).to_deterministic()


    #add margins to masks
    temp_mask = np.zeros((masks.shape[0], masks.shape[1], num_classes))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 3x3 rectangular kernel
    eroded_channels = []
    for i in range(num_classes):
        temp_mask[:,:,i][masks == i] = masks[masks == i]
        eroded_channel = cv2.erode(temp_mask[:, :, i], kernel, iterations=3)
        eroded_channels.append(eroded_channel)


    # Stack the channels back together
    eroded_image = np.stack(eroded_channels, axis=-1)
    masks = np.sum(eroded_image, axis=2)
    i_m[:, :, 0:3] = np.copy(images)
    i_m[:, :, 3] = np.copy(masks)
    i_m = ag1.augment_image(i_m)


    images1 = i_m[:, :, 0:3].astype('float32')

    masks1 = i_m[:, :, 3].astype('int16')



    return images1, masks1



class PumaTissueDataset(Dataset):
    def __init__(self,
                 imgs1,
                 masks,
                 n_class1 = 6,
                 size1 = (128,128),
                 device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 transform = None,
                 mode="train",
                 target_size = (1024,1024),
                 train_indexes = None,
                 er_di = False
                 ):
        self.imgs = imgs1
        self.masks = masks
        self.n_class = n_class1
        self.image = torch.zeros((3,size1[0],size1[1]), device=device1)
        self.mask = torch.zeros((n_class1,size1[0],size1[1]), device=device1)
        self.size = size1
        self.target_size = target_size
        # self.resizeM = Resize(size = size1, interpolation=InterpolationMode.NEAREST)
        # self.resizeI = Resize(size1)
        self.device = device1
        self.transform = transform
        self.mode = mode
        self.train_indexes = train_indexes
        self.er_di = er_di
        if transform is not None:
            a = 0
            # self.setup_augmentor(0)
        return

    def setup_augmentor(self, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        return



    def __len__(self):
        if (self.train_indexes is not None) and (self.mode == "train"):
            lent = len(self.train_indexes)
        else:
            lent = len(self.imgs)
        return lent

    def __getitem__(self, idx):

        if (self.train_indexes is not None) and (self.mode == "train"):
            idx = int(self.train_indexes[idx])
        image = self.imgs[idx].astype("float32")
        temp = self.masks[idx].astype("int16")

        if self.transform is not None:
            image[:,:,0:3] = distort_image_with_randaugment(image[:,:,0:3])

        if self.er_di:
            disk_radius = 5
            kernel_size = (2 * disk_radius + 1, 2 * disk_radius + 1)

            # Create the circular disk structuring element
            disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            eroded_image = cv2.erode(image, disk_kernel, iterations=1)

            # Apply dilation
            dilated_image = cv2.dilate(image, disk_kernel, iterations=1)
            image = np.concatenate((image,eroded_image, dilated_image), axis=2)


        image = image/255
        image = np.transpose(image, (2, 0, 1))
        temp = temp.astype("int32")

        # if np.sum(temp > self.n_class-1) or np.sum(temp < 0):
        #     temp[temp > self.n_class-1] = 0
        #     temp[temp < 0] = 0

        image = image.astype('float32')
        image = np.copy(image)
        temp = np.copy(temp)

#        image1 = self.image
#        image1[:,:,:] = ToTensor()(np.transpose(image,(2,0,1)))
#        temp = ToTensor()(temp)
#         print(idx)




        return image, temp, idx





    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding

                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.size[0], self.size[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
                # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25), seed=rng),
                # iaa.ElasticTransformation(alpha=(0.0, 2.0), sigma=0.25, seed=rng),

            ]

            input_augs = [
                # iaa.Multiply((0.5, 1.5), per_channel=0.2),
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                        iaa.GaussianBlur(sigma=(0.0, 1.0),seed=rng),
                        # iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30),seed=rng),
                        # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                        # iaa.MultiplySaturation((0.5, 1.5)),

                    ]
                ),
                iaa.Sequential(
                    [

                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs

