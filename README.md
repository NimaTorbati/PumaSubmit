# A Multi-Stage Auto-Context Deep Learning Framework for Tissue and Nuclei Segmentation and Classification in H&E-Stained Histological Images of Advanced Melanoma
This repository contains the LSM team's code developed for the Panoptic Segmentation of Nuclei and Tissue in Advanced Melanoma (PUMA) Challenge 
<br/>[Puma Challenge Website](https://puma.grand-challenge.org/#panoptic-segmentation-of-nuclei-and-tissue-in-advanced-melanoma)
<br/>[Our Model weights](https://huggingface.co/datasets/NiToLSM/PumaWeightsNiTo_LSM)
<br/>[Puma Dataset Paper](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf011/8024182?login=false)
<br/>Acknowledgment:
This project has been conducted through a joint WWTF-funded project (Grant ID: 10.47379/LS23006) between the Medical University of Vienna and Danube Private University.

# Inference
An example of how to run inference for 10 nuclei classes is shown in inference_offline.py:

1. Download the model weights from the HuggingFace link and place them in: 'Docker/DockerTrack2/inference/'.

2. Download HoverNext weights 'Hover-NeXt_all_classes' from: https://zenodo.org/records/13881999.
   Place the 'best_model' file inside:
   'Docker/DockerTrack2/checkpoint/train/'.
3. Update the image and prediction directory paths in inference_offline.py based on your operating system.
4. Run inference_offline.py.
# Model Stages
Stage 1: <img width="3100" height="1088" alt="stage1" src="https://github.com/user-attachments/assets/22003e0a-04cf-4b31-8e31-3d53966d641e" />

# Citation
A preprint version of our paper is publicy available at: 
<br/>[Paper](https://arxiv.org/abs/2503.23958)
