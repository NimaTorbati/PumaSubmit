#!/usr/bin/env bash


# Step 1: perform tissue inference

cd ./inference
python inference_4th_submission.py
#python inference_cll_tissue.py
python inference_cell_tissue_unetpp10.py
python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py
#python inference_tissue_from_nuclei.py
#python inference_cell_tissue_unet.py

cd ../


# Step 2: perform nuclei inference with hovernext
python process.py "$@"


