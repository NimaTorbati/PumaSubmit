import numpy as np
np.bool=np.bool_
from utils import compute_puma_dice_micro_dice,calculate_micro_dice_score_with_masks
import numpy as np
np.bool=np.bool_
import numpy as np
from utils import copy_data
np.bool = np.bool_
import torch
from sklearn.model_selection import KFold
import os
from pathlib import Path
import shutil
from typing import List
import json
from shapely.geometry import shape
from rasterio.features import rasterize
import tifffile
from transformers import SegformerForSemanticSegmentation,SegformerConfig


if __name__ == '__main__':
    tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']

    tissue_images_path ='/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_tif_ROIs/'
    tissue_labels_path = '/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_geojson_tissue/'
    final_target_size = (1024,1024)
    classifier_n_class = 11
    n_class = 6
    model_name = 'segformer'
    classifier_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/inference/Model_weights/foldAllSegformer0'
    primary_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/third_submission_weights/foldRawPrimary'
    metastatic_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/third_submission_weights/foldRawMetastatic'

    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3

    val_percent = 0.2


    image_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims.npy')
    mask_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/masks.npy')




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


    true_cases = 0
    all_cases = 0
    for folds in range(n_folds):
        print('training fold ', str(folds))
        val_index_primary = splits_primary[folds][1]
        val_index_metas = splits_metas[folds][1]



        # val_images = image_data_metas[val_index_metas]# np.concatenate((image_data_metas[val_index_metas],image_data_primary[val_index_primary]),axis=0)
        # val_masks = mask_data_metas[val_index_metas]#np.concatenate((mask_data_metas[val_index_metas], mask_data_primary[val_index_primary]), axis=0)

        val_images = np.concatenate((image_data_metas[val_index_metas],image_data_primary[val_index_primary]),axis=0)
        val_masks = np.concatenate((mask_data_metas[val_index_metas], mask_data_primary[val_index_primary]), axis=0)


        val_save_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth/foldFullEval' + str(folds)
        val_save_path1 = '/home/ntorbati/PycharmProjects/pythonProject/validation_images/foldFullEval' + str(folds)
        output_folder = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/foldFullEval' + str(folds)


        copy_data(validation_indices = val_index_primary, data_path = tissue_labels_path,data_path1= tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'primary')
        # copy_data(validation_indices=[], data_path=tissue_labels_path, data_path1=tissue_images_path,
        #           save_path=val_save_path, save_path1=val_save_path1, data_type='primary')
        copy_data(validation_indices = val_index_metas, data_path = tissue_labels_path,data_path1=tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'metastatic')



        #classifier_model
        num_input_channels = 3  # for RGB images
        num_output_channels = 11  # for 6 segmentation classes
        dir_checkpoint = Path(
            classifier_path + str(
                folds) + '/')

        PATH = str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1))
        # Load the same pretrained configuration to maintain architecture consistency
        config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

        # Modify the configuration to match your dataset
        config.num_labels = num_output_channels  # Set the number of segmentation classes
        config.image_size = 1024  # Ensure input image size is 1024x1024

        # Initialize the model (without pretrained weights)
        model_classifier = SegformerForSemanticSegmentation(config)

        # Load the fine-tuned weights
        state_dict = torch.load(PATH, weights_only=True)  # Load on CPU for safety
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Load the state dict with strict=False to ignore minor mismatches
        model_classifier.load_state_dict(state_dict)  # strict=True to ensure all trained layers match
        model_classifier.to(device2)
        model_classifier.eval()


        #segmentation model
        #classifier_model
        num_input_channels = 3  # for RGB images
        num_output_channels = 6  # for 6 segmentation classes
        # Load the same pretrained configuration to maintain architecture consistency
        config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

        # Modify the configuration to match your dataset
        config.num_labels = num_output_channels  # Set the number of segmentation classes
        config.image_size = 1024  # Ensure input image size is 1024x1024

        # Initialize the model (without pretrained weights)
        model_segment = SegformerForSemanticSegmentation(config)
        model_segment.to(device2)
        model_segment.eval()

        # Load the fine-tuned weights



        if os.path.exists(output_folder):
            for root, dirs, files in os.walk(output_folder, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(output_folder)
        os.makedirs(output_folder, exist_ok=True)



        with torch.no_grad():
            for file_name in os.listdir(val_save_path1):
                if file_name.endswith(".tif"):
                    model_classifier.n_classes = classifier_n_class

                    target_size = final_target_size
                    size = target_size[0]




                    _,primary_sum,metas_sum = compute_puma_dice_micro_dice(model=model_classifier,
                                                            target_siz=final_target_size,
                                                            epoch=0,
                                                            input_folder=val_save_path1,
                                                            output_folder=output_folder,
                                                            ground_truth_folder=val_save_path,
                                                            device=device2,
                                                            save_jpg=True,
                                                            file_path=file_name,
                                                              classifier_mode=True

                                                            )
                    if primary_sum[0][3] > 0:
                        seg_path = primary_path
                        key = 'primary'
                    elif np.sum(primary_sum) > np.sum(metas_sum):
                        seg_path = primary_path
                        key = 'primary'
                    else:
                        seg_path = metastatic_path
                        key = 'metastatic'
                    if key in file_name:
                        true_cases += 1
                    else:
                        print(file_name)
                        print(key, ' but classifier failed: primary = ', primary_sum , 'metastatic = ', metas_sum)
                    all_cases += 1



                    dir_checkpoint = Path(
                        seg_path + str(
                            folds) + '/')

                    PATH = str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1))

                    state_dict = torch.load(PATH, weights_only=True)  # Load on CPU for safety
                    if "module." in list(state_dict.keys())[0]:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    # Load the state dict with strict=False to ignore minor mismatches
                    model_segment.load_state_dict(state_dict)  # strict=True to ensure all trained layers match

                    prediction,_,_ = compute_puma_dice_micro_dice(model=model_segment,
                                                            target_siz=final_target_size,
                                                            epoch=0,
                                                            input_folder=val_save_path1,
                                                            output_folder=output_folder,
                                                            ground_truth_folder=val_save_path,
                                                            device=device2,
                                                            save_jpg=True,
                                                            file_path=file_name
                                                            )
            micro_dices, mean_micro_dice = calculate_micro_dice_score_with_masks(val_save_path, output_folder,
                                                                                 (1024,1024), eps=0.00001)
            print(mean_micro_dice)
    print('classifier_acc = ' ,true_cases/all_cases)
