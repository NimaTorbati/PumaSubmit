from ptflops import get_model_complexity_info
import numpy as np
from hoverNext.multi_head_unet import get_model as get_hovernext
np.bool=np.bool_
import segmentation_models_pytorch as smp
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
import pandas as pd


RESOLUTION = 1024


def flopsEstimate(
    model,
    in_channels=3,
):
    model.to('cuda')
    model.eval()

    # macs
    macs, params = get_model_complexity_info(
        model,
        (in_channels, RESOLUTION, RESOLUTION),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )

    # FLOPs
    flops_per_image_inference = 2 * macs






    print("\nMODEL STATS")
    print(f"Parameters: {params/1e6:.2f} M")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")

    print("\n- FLOPs -")
    print(f"MACs per image: {macs/1e9:.2f} GMACs")
    print(f"Inference FLOPs per image: {flops_per_image_inference/1e9:.2f} GFLOPs")

    return {
        "params": params,
        "macs_per_image": macs,
    }





if __name__ == "__main__":

    in_channels = 3
    num_samples = 205


    num_output_channels = 6
    config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    num_input_channels = in_channels
    config.num_channels = num_input_channels
    config.num_labels = num_output_channels
    config.image_size = 1024

    # Initialize the model (without pretrained weights)
    model1 = SegformerForSemanticSegmentation(config)

    # model1 = smp.Unet(classes=6, in_channels=in_channels)

    # model1 = smp.UnetPlusPlus(
    #     encoder_name="resnet50",
    # )


    # model1 = get_hovernext(out_channels_cls=1,
    #                       out_channels_inst=5,
    #                       pretrained=True, )

    flopsEstimate(
        model=model1,
        in_channels=3,
    )
