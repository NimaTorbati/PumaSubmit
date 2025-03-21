#!/usr/bin/env bash


# Step 1: initial tissue segmentation

cd ./inference
python inference_4th_submission.py
#python inference_cll_tissue.py

# Step 2: Nuclei segmentation

python inference_cell_tissue_unetpp.py
#python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py
#python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py

cd ../


# Step 3: perform nuclei inference with hovernext
python process.py "$@"

# Step 4: perform tissue segmentation from nuclei
cd ./inference
python inference_cell_tissue_unetpp10.py
python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py
#python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py

cd ../

