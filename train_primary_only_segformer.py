import numpy as np
np.bool=np.bool_
import segmentation_models_pytorch as smp
import torch
from train_puma_dice import train_model
from sklearn.model_selection import KFold
from pathlib import Path
from utils import Data_class_analyze,modify_model_for_new_classes,fill_background_holes_batch,copy_data
from torch import nn
import cv2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
if __name__ == '__main__':
    tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']

    # final_target_size = (1024,1024)
    tissue_images_path ='/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_tif_ROIs/'
    tissue_labels_path = '/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_geojson_tissue/'
    #
    # all_tissue_data = np.sort([tissue_images_path + image for image in os.listdir(tissue_images_path)])
    # all_tissue_labels = np.sort([tissue_labels_path + labels for labels in os.listdir(tissue_labels_path)])
    # image_data, mask_data = load_data_tissue(target_size = final_target_size,
    #                                   data_path = all_tissue_data,
    #                                   annot_path = all_tissue_labels,
    #                                   tissue_labels = tissue_labels,
    #                                   im_size = [1024,1024])
    final_target_size = (1024,1024)
    n_class = 6
    fine_tune = False
    use_necros = False
    progressive = False
    parallel = False

    if fine_tune:
        n_class_old = 6
        n_class_new = n_class
        fineTune_PATH = "/home/ntorbati/PycharmProjects/pythonProject/E:/PumaDataset/checkpoints/RawMetasPanoptSameclassesSegFormer/check.pth"

    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3

    val_percent = 0.2

    if use_necros:
        image_data_necros = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/necros_ims.npy')
        image_data_necros = image_data_necros[0:100]
        mask_data_necros = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/necros_masks.npy')
        mask_data_necros = mask_data_necros[0:100]
        mask_data_necros = fill_background_holes_batch(mask_data_necros)

        res = 880
        image_data_necros = image_data_necros[:, 0:res, 0:res, :]
        mask_data_necros = mask_data_necros[:, 0:res, 0:res]

        image_data_necros = np.array([cv2.resize(img, final_target_size, interpolation=cv2.INTER_LINEAR) for img in image_data_necros])

        mask_data_necros = np.array([cv2.resize(mask, final_target_size, interpolation=cv2.INTER_NEAREST) for mask in mask_data_necros])
        # mask_data_necros[mask_data_necros == 5] = 3

    image_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims.npy')
    mask_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/masks.npy')


    # mask_data, tissue_labels, _ = add_small_labels(mask_data, tissue_labels)
    # tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']


    # image_data_metas = image_data[0:102]
    # mask_data_metas = mask_data[0:102]


    image_data_primary = image_data[103:]
    mask_data_primary = mask_data[103:]


    image_data = np.empty(0)
    mask_data = np.empty(0)







    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # indices_metas = np.arange(image_data_metas.shape[0])
    indices_primary = np.arange(image_data_primary.shape[0])


    # splits_metas = list(kf.split(indices_metas))
    splits_primary = list(kf.split(indices_primary))


    for folds in range(0,4):#n_folds):
        print('training fold ', str(folds))
        train_index_primary = splits_primary[folds][0]
        val_index_primary = splits_primary[folds][1]
        print(val_index_primary)

        # train_index_metas = splits_metas[folds][0]
        # val_index_metas = splits_metas[folds][1]





        val_images = image_data_primary[val_index_primary]#np.concatenate((image_data_metas[val_index_metas],image_data_primary[val_index_primary]),axis=0)
        val_masks = mask_data_primary[val_index_primary]#np.concatenate((mask_data_metas[val_index_metas], mask_data_primary[val_index_primary]), axis=0)
        print('necros area = ', np.sum(np.where(val_masks == 5)))
        train_images = image_data_primary[train_index_primary]#np.concatenate((image_data_metas[train_index_metas],image_data_primary[train_index_primary]),axis=0)##
        train_masks = mask_data_primary[train_index_primary]#np.concatenate((mask_data_metas[train_index_metas], mask_data_primary[train_index_primary]), axis=0)##

        if use_necros:
            train_images = np.concatenate((train_images,image_data_necros),axis=0)
            train_masks = np.concatenate((train_masks,mask_data_necros),axis=0)

        # train_masks[train_masks == 5] = 3
        train_indexes = np.linspace(0, len(train_images)-1, len(train_images))
        # train_indexes = np.append(train_indexes, diff_cases)

        dir_checkpoint = Path('E:/PumaDataset/checkpoints/foldRawPrimary1Segformer' + str(folds) + '/')
        val_save_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth/fold1' + str(folds)
        val_save_path1 = '/home/ntorbati/PycharmProjects/pythonProject/validation_images/fold1' + str(folds)
        output_folder = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/fold1' + str(folds)


        copy_data(validation_indices = val_index_primary, data_path = tissue_labels_path,data_path1= tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'primary')
        copy_data(validation_indices = [], data_path = tissue_labels_path,data_path1=tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'metastatic')


        # train_images, train_masks = blood_vessel_aug(train_masks = train_masks, train_images = train_images, num_classes = n_class)

        class_weights = [1, 1.5, 2.2, 1.4, 4,1]
        class_weights = torch.tensor(class_weights, device=device2,dtype=torch.float16)
        iters = [400]
        lr = 1e-5

        scals = 0


        if fine_tune:
            model1 = smp.Unet(classes=n_class_old)
            model1.load_state_dict(torch.load(fineTune_PATH, weights_only=True))
            model1 = modify_model_for_new_classes(model=model1,n_classes=n_class_new)
        else:
            # model1 = smp.Unet(classes=n_class)
            # PATH = "/home/ntorbati/PycharmProjects/pythonProject/E:/PumaDataset/checkpoints/RawMetasPanopt5/check.pth"
            # model1.load_state_dict(torch.load(PATH, weights_only=True))
            num_input_channels = 3  # for RGB images
            num_output_channels = 6  # for 6 segmentation classes
            # fineTune_PATH = "/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/third_submission_weights/foldRawPrimary" + str(folds) + "/checkpoint_epoch1.pth"
            fineTune_PATH = "/home/ntorbati/PycharmProjects/pythonProject/E:/PumaDataset/checkpoints/RawMetasPanoptSameclassesSegFormer/check1.pth"

            # Load the same pretrained configuration to maintain architecture consistency
            config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

            # Modify the configuration to match your dataset
            config.num_labels = num_output_channels  # Set the number of segmentation classes
            config.image_size = 1024  # Ensure input image size is 1024x1024

            # Initialize the model (without pretrained weights)
            model1 = SegformerForSemanticSegmentation(config)

            # Load the fine-tuned weights
            state_dict = torch.load(fineTune_PATH, map_location=torch.device('cpu'))  # Load on CPU for safety

            # Load the state dict with strict=False to ignore minor mismatches
            model1.load_state_dict(torch.load(fineTune_PATH, weights_only=True))  # strict=True to ensure all trained layers match




        if parallel:
            model1 = nn.DataParallel(model1)
        model1.to(device2)
        #    summary(model1, input_size=(in_channels, size, size))
        model1.n_classes = n_class

        target_size = final_target_size
        size = target_size[0]

        model1 = train_model(
        model = model1,
        device = device2,
        epochs = iters[scals],
        batch_size = 3,
        learning_rate = lr,
        val_percent = 0.2,
        save_checkpoint = True,
        img_scale = 0.5,
        amp = False,
        weight_decay=0.7,  # learning rate decay rate
        momentum = 0.999,
        gradient_clipping = 1.0,
        target_siz=target_size,
        n_class=n_class,
        image_data1=train_images,
        mask_data1=train_masks,
        val_images = val_images,
        val_masks = val_masks,
        class_weights = class_weights,
        augmentation=True,# default None
        val_batch=1,
        early_stopping=100,
        ful_size=final_target_size,
            val_augmentation=True,
            train_indexes = train_indexes,
            input_folder=val_save_path1,
            output_folder=output_folder,
            ground_truth_folder=val_save_path,
            folds = n_folds,
            dir_checkpoint=dir_checkpoint,
            logg=True,
            progressive=progressive,
            model_name='segformer',
            val_sleep_time=300
            # grad_wait=int(20 / 10)
        )
