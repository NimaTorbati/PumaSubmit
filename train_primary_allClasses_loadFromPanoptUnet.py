import numpy as np
np.bool=np.bool_
import segmentation_models_pytorch as smp
import torch
from train_puma_dice import train_model
from sklearn.model_selection import KFold
from pathlib import Path
from utils import Data_class_analyze,copy_data
from torch import nn

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
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3

    val_percent = 0.2

    image_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims.npy')
    mask_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/masks.npy')


    # mask_data, tissue_labels, _ = add_small_labels(mask_data, tissue_labels)
    # tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']


    image_data_metas = image_data[0:102]
    mask_data_metas = mask_data[0:102]


    image_data_primary = image_data[103:]
    mask_data_primary = mask_data[103:]


    image_data = np.empty(0)
    mask_data = np.empty(0)







    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    indices_metas = np.arange(image_data_metas.shape[0])
    indices_primary = np.arange(image_data_primary.shape[0])


    splits_metas = list(kf.split(indices_metas))
    splits_primary = list(kf.split(indices_primary))


    for folds in range(3,n_folds):
        print('training fold ', str(folds))
        train_index_primary = splits_primary[folds][0]
        val_index_primary = splits_primary[folds][1]
        print(val_index_primary)

        train_index_metas = splits_metas[folds][0]
        val_index_metas = splits_metas[folds][1]





        val_images = np.concatenate((image_data_metas[val_index_metas],image_data_primary[val_index_primary]),axis=0)
        val_masks = np.concatenate((mask_data_metas[val_index_metas], mask_data_primary[val_index_primary]), axis=0)

        train_images = np.concatenate((image_data_metas[train_index_metas],image_data_primary[train_index_primary]),axis=0)
        train_masks = np.concatenate((mask_data_metas[train_index_metas], mask_data_primary[train_index_primary]), axis=0)




        train_indexes = np.linspace(0, len(train_images)-1, len(train_images))
        # train_indexes = np.append(train_indexes, diff_cases)

        dir_checkpoint = Path('E:/PumaDataset/checkpoints/foldRawPrimary2unet' + str(folds) + '/')
        val_save_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth/fold1unet' + str(folds)
        val_save_path1 = '/home/ntorbati/PycharmProjects/pythonProject/validation_images/fold1unet' + str(folds)
        output_folder = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/fold1unet' + str(folds)


        copy_data(validation_indices = val_index_primary, data_path = tissue_labels_path,data_path1= tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'primary')
        copy_data(validation_indices = val_index_metas, data_path = tissue_labels_path,data_path1=tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'metastatic')


        # train_images, train_masks = blood_vessel_aug(train_masks = train_masks, train_images = train_images, num_classes = n_class)

        class_weights = [1,4, 10, 4, 6, 4]
        class_weights = torch.tensor(class_weights, device=device2,dtype=torch.float16)
        iters = [50]
        lr = 1e-5

        scals = 0
        parallel = False
        model1 = smp.Unet(classes=n_class)
        PATH = "/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/inference/Model_weights/foldRawPrimary" + str(folds) + "/checkpoint_epoch1.pth"
        model1.load_state_dict(torch.load(PATH, weights_only=True))
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
        batch_size = 5,
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
            # grad_wait=int(20 / 10)
        )
