from src.post_process_utils import (
    work,
    write,
    get_pp_params,
    get_shapes,
    get_tile_coords,
)
import tifffile
from src.data_utils import NpyDataset, ImageDataset
from typing import List, Tuple
import zarr
from numcodecs import Blosc
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import json
import os
from typing import Union
from tqdm.auto import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import Counter
def post_process_main(
    params: dict,
    z: Union[Tuple[zarr.ZipStore, zarr.ZipStore], None] = None,
):
    """
    Post processing function for inference results. Computes stitched output maps and refines prediction results and produces instance and class maps

    Parameters
    ----------

    params: dict
        Parameter store, defined in initial main

    """
    # get best parameters for respective evaluation metric
    params = get_pp_params(params, True)
    params, ds_coord = get_shapes(params, len(params["best_fg_thresh_cl"]))

    tile_crds = get_tile_coords(
        params["out_img_shape"],
        params["pp_tiling"],
        pad_size=params["pp_overlap"],
        npy=params["input_type"] != "wsi",
    )
    pinst_out = zarr.zeros(
            shape=(params["orig_shape"][0], *params["orig_shape"][-2:]),
            dtype="i4",
            compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
        )

    executor = ProcessPoolExecutor(max_workers=params["pp_workers"])
    tile_processors = [
        executor.submit(work, tcrd, ds_coord, z, params) for tcrd in tile_crds
    ]
    pcls_out = {}
    running_max = 0

    for future in tqdm(
        concurrent.futures.as_completed(tile_processors), total=len(tile_processors)
    ):
        pinst_out, pcls_out, running_max = write(
            pinst_out, pcls_out, running_max, future.result(), params
        )
    executor.shutdown(wait=False)

    # Create the final JSON output from the processed class map
    output_path = params["root"] + params["output"]
    if output_path is not None:
        print("storing class dictionary as JSON...")
        output_json = create_polygon_json_ours(pinst_out, pcls_out, params)
        # output_json = create_polygon_json(pinst_out, pcls_out, params)
        # add version (required by GC)
        output_json["version"] = {
            "major": 1,
            "minor": 0
        }

        # save JSON file
        json_filename = os.path.join('/output/melanoma-3-class-nuclei-segmentation.json')
        # json_filename = os.path.join(os.path.dirname(params["output_dir"]), 'melanoma-3-class-nuclei-segmentation.json')
        with open(json_filename, "w") as fp:
            json.dump(output_json, fp, indent=2)
        print(f"JSON file saved to {json_filename}")
        # Debug: Verify file was saved
        if os.path.exists(json_filename):
            print(f"Successfully saved: {json_filename}")
        else:
            print(f"Failed to save: {json_filename}")


def create_polygon_json(pinst_out, pcls_out, params):
    """
    Converts the instance map and class map into a JSON structure for polygon output.

    Parameters
    ----------
    pinst_out: zarr array
        In-memory instance segmentation results.
    pcls_out: dict
        Class map containing instance-to-class mapping information.
    params: dict
        Parameter store, defined in initial main.

    Returns
    ----------
    output_json: dict
        JSON structure containing polygon data.
    """
    # load instance map (2D map of instance IDs)
    full_instance_map = np.squeeze(pinst_out, axis=0)
    nuc_path = params['nuclei_pred_path']
    inst_path = str(f"{nuc_path}").replace(".tif", "_inst.npy")
    np.save(inst_path, full_instance_map)

    # Create a class map using instance IDs from pcls_out
    pcls_list = np.array([0] + [v[0] for v in pcls_out.values()])  # Class IDs
    pcls_keys = np.array(["0"] + list(pcls_out.keys())).astype(int)  # Instance IDs
    lookup = np.zeros(pcls_keys.max() + 1, dtype=np.uint8)  # Lookup table
    lookup[pcls_keys] = pcls_list
    class_map = lookup[full_instance_map]
    # Define colors and class names for the different classes
    colors = {
        1: (0, 255, 0),  # lymphocytes
        2: (255, 0, 0),  # tumor
        3: (0, 0, 255)   # other
    }
    class_names = {
        1: 'nuclei_lymphocyte',
        2: 'nuclei_tumor',
        3: 'nuclei_other'
    }

    output_json = {
        "type": "Multiple polygons",
        "polygons": []
    }

    for class_id in range(1, 4):
        # create binary mask for current class
        class_mask = (class_map == class_id).astype(np.uint8)

        # find outlines of binary mask
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # convert contour points to the desired format (x, y, z)
            # note: since we are dealing with 2D-polygons, we choose to set Z to 0.5
            path_points = [[float(pt[0][0]), float(pt[0][1]), 0.5] for pt in contour]

            # create a polygon entry
            polygon_entry = {
                "name": class_names[class_id],
                "seed_point": path_points[0],  # using first point as the seed point
                "path_points": path_points,
                "sub_type": "",  # empty string for subtype
                "groups": [],  # empty array for groups
                "probability": 1  # confidence score
            }
            output_json["polygons"].append(polygon_entry)

    return output_json

def create_polygon_json_ours(pinst_out, pcls_out, params):
    """
    Converts the instance map and class map into a JSON structure for polygon output.

    Parameters
    ----------
    pinst_out: zarr array
        In-memory instance segmentation results.
    pcls_out: dict
        Class map containing instance-to-class mapping information.
    params: dict
        Parameter store, defined in initial main.

    Returns
    ----------
    output_json: dict
        JSON structure containing polygon data.
    """
    # load instance map (2D map of instance IDs)
    full_instance_map = np.squeeze(pinst_out, axis=0)

    # Create a class map using instance IDs from pcls_out

    # nuc_path = params['nuclei_pred_path']
    nuc_path = params['p1'].replace(".tif", ".npy")
    # inst_path = str(f"{nuc_path}").replace(".tif", "_inst.npy")
    # np.save(inst_path, full_instance_map)
    class_map = np.load(nuc_path)

    # class_map = cv2.cvtColor(cv2.imread(nuc_path), cv2.COLOR_BGR2GRAY)

    class_map = merge_instance_maps(full_instance_map, class_map,params)
    # Define colors and class names for the different classes
    colors = {
        1: (0, 255, 0),  # lymphocytes
        2: (255, 0, 0),  # tumor
        3: (0, 0, 255)   # other
    }
    class_names = {
        1: 'nuclei_lymphocyte',
        2: 'nuclei_tumor',
        3: 'nuclei_other'
    }

    output_json = {
        "type": "Multiple polygons",
        "polygons": []
    }

    for class_id in range(1, len(class_names)+1):
        # create binary mask for current class
        class_mask = (class_map == class_id).astype(np.uint8)

        # find outlines of binary mask
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # convert contour points to the desired format (x, y, z)
            # note: since we are dealing with 2D-polygons, we choose to set Z to 0.5
            path_points = [[float(pt[0][0]), float(pt[0][1]), 0.5] for pt in contour]

            # create a polygon entry
            polygon_entry = {
                "name": class_names[class_id],
                "seed_point": path_points[0],  # using first point as the seed point
                "path_points": path_points,
                "sub_type": "",  # empty string for subtype
                "groups": [],  # empty array for groups
                "probability": 1  # confidence score
            }
            output_json["polygons"].append(polygon_entry)

    return output_json


def merge_instance_maps(full_instance_map, class_map,params):
    # Label connected components in the instance map
    full_instance_map = fill_margins(full_instance_map1=full_instance_map, class_map1=class_map)

    labeled_instances, num_features = label(full_instance_map)

    # Create an output map initialized with zeros (background)
    output_map = np.zeros_like(full_instance_map, dtype=np.uint8)

    for instance_id in range(1, num_features + 1):
        # Get the mask of the current instance
        instance_mask = labeled_instances == instance_id

        # Get the corresponding class values from the class_map
        class_values = class_map[instance_mask]

        # Perform majority voting to determine the most common class
        if np.sum(class_values>0):
            class_values = class_values[class_values>0]

        # if np.sum(class_values==2):
        #     class_values = class_values[class_values>0]
        #     if np.sum(class_values==3):
        #         class_values = class_values[class_values == 3]
                # if np.sum(class_values==3):
                #     class_values = class_values[class_values ==3]


        class_counts = Counter(class_values)
        most_common_class, _ = class_counts.most_common(1)[0]

        # If the majority class is 0 (background), assign class 3
        # if most_common_class == 0:
        #     most_common_class = 3

        # Assign the determined class to the output map
        output_map[instance_mask] = most_common_class
    plt.imshow(output_map)
    plt.show()
    return output_map


def fill_margins(full_instance_map1 =None, class_map1 = None):
    full_instance_map2 = np.copy(full_instance_map1)
    class_map2 = np.copy(class_map1)
    full_instance_map2[full_instance_map2>0] = 1
    class_map2[class_map2>0] = 1
    margin_size = full_instance_map2.shape[0] // 32

    # Extract margin regions from class_map
    top_margin = class_map2[:margin_size, :]
    bottom_margin = class_map2[-margin_size:, :]
    left_margin = class_map2[:, :margin_size]
    right_margin = class_map2[:, -margin_size:]

    # Fill the margins in full_instance_map
    full_instance_map2[:margin_size, :] = top_margin
    full_instance_map2[-margin_size:, :] = bottom_margin
    full_instance_map2[:, :margin_size] = left_margin
    full_instance_map2[:, -margin_size:] = right_margin

    return full_instance_map2