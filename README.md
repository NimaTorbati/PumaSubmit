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
Stage 1: 'The first stage of the proposed method. In this stage, a SegFormer model is
trained to classify the input image type (primary or metastatic) based on segmentation
result and the classification rules.' <img width="3100" height="1088" alt="stage1" src="https://github.com/user-attachments/assets/22003e0a-04cf-4b31-8e31-3d53966d641e" />

Stage 2: 'The second stage of the proposed method. Initial tissue segmentation is per-
formed using two SegFormer models, one for primary tissues and the other for metastatic
tissues. Due to the Segformer model’s poor performance in segmenting blood vessels, a
U-Net model is trained separately for blood vessel detection and the results are refined
using the tissue ensemble rules. Color code for tissue segmentation: red for tumor, blue
for stroma, and green for blood vessel.' <img width="2274" height="897" alt="stage2" src="https://github.com/user-attachments/assets/5f642172-bb4b-4baf-8eeb-447889338e41" />

Stage 3: 'The third stage of the proposed method. This stage consists of two branches,
with the upper part dedicated to nuclei classification and the lower part to nuclei instance
segmentation. In the first branch, a U-Net++ model is trained for nuclei class map de-
tection, incorporating tissue segmentation results as an additional input channel. In the
second branch, nuclei instance segmentation is performed using the HoVer-NeXt model.
The final nuclei segmentation and classification is obtained by combining the class maps
and instance masks through a majority voting approach. Color code for tissue segmen-
tation: red for tumor, blue for stroma, and green for blood vessel; color code for nuclei
segmentation: various cell types are indicated by different colors.' <img width="2682" height="843" alt="stage3" src="https://github.com/user-attachments/assets/47fbb2bd-6c2c-460e-842f-3ded069602a3" />

Stage 4: 'The fourth and final stage of the proposed method. Tissue segmentation is re-
fined using nuclei segmentation results. The nuclei segmentation output from the previous
stage is added as an additional input channel, and two SegFormer models similar to those
in stage 2 are trained. For the U-Net model, we used the same trained model in stage 2
without incorporating the fourth nuclear channel. Color code for tissue segmentation: red
for tumor, blue for stroma, and green for blood vessel; color code for nuclei segmentation:
various cell types are indicated by different colors.' <img width="5625" height="1300" alt="stage4" src="https://github.com/user-attachments/assets/c69da807-8f99-46cc-8bf9-478d79ba72b6" />

# Citation
A preprint version of our paper is publicy available at: 
<br/>[Paper](https://arxiv.org/abs/2503.23958)
