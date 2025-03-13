#!/usr/bin/env bash


# Step 1: perform tissue inference with nnunet

cd ./inference
python inference_4th_submission.py
#python inference_cll_tissue.py
python inference_cell_tissue_unetpp.py
#python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py
#python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py

cd ../


# Step 2: perform nuclei inference with hovernext
python process.py "$@"

cd ./inference
python inference_cell_tissue_unetpp10.py
python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py
#python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py

cd ../

# create the original name and nnunet name list
#python create_convert_dict.py
# reload tiff image into png and change name
#python image_transfer.py
# inference


#cd ./nnunetv2/inference/
#python predict_from_raw_data.py
# convert gt json file into png with correspond name
#cd ../../
#python output_rename.py

