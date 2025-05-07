import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils

from detectron2.structures import BoxMode

from utils import read_maskfile, encode_mask

def get_train_dicts():

    """
    data/
      train/
        <image_id>/
          image.tif
          class1.tif (optional)
          class2.tif (optional)
          ...
    return: Detectron2 annotation dict list。
    """
    dataset_dicts = []
    train_dir = os.path.join("data", "train")
    for img_id in tqdm(os.listdir(train_dir), total=len(os.listdir(train_dir))):
        img_folder = os.path.join(train_dir, img_id)
        img_path = os.path.join(img_folder, "image.tif")
        if not os.path.exists(img_path):
            continue

        # read the image and get its height and width
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        record = {
            "file_name": img_path,
            "image_id": img_id,
            "height": height,
            "width": width,
            "annotations": []
        }

        # read the mask of class1 ~ class4 
        for class_idx in range(1, 5):
            mask_path = os.path.join(img_folder, f"class{class_idx}.tif")
            if not os.path.exists(mask_path):
                continue

            mask = read_maskfile(mask_path)
            max_value = int(np.max(mask))

            for pixel_value in range(1, max_value+1):
            
                # get the binary mask
                bin_mask = (mask == pixel_value)
                # rle format
                rle = encode_mask(bin_mask)
            
                area = float(mask_utils.area(rle))
                bbox = mask_utils.toBbox(rle).tolist()
                if area > 1:
                    record["annotations"].append({
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": rle.copy(),
                        "area": area,
                        "category_id": class_idx - 1
                    })

        dataset_dicts.append(record)

    return dataset_dicts


def get_test_dicts():

    """
    data/
      test_release/
        <image_id>.tif
        ...
    
    return: Detectron2 annotation dict list。
    """
    dataset_dicts = []
    test_dir = os.path.join("data", "test_release")
    for img_filename in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_filename)
        img_id = img_filename.split(".")[0]
        if not os.path.exists(img_path):
            continue

        # read the image and get its height and width
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        record = {
            "file_name": img_path,
            "image_id": img_id,
            "height": height,
            "width": width,
            "annotations": []
        }

        dataset_dicts.append(record)

    return dataset_dicts

def get_data_by_indices(data, indices):
    return [
        data[idx] for idx in indices
    ]

import pickle

if __name__ == "__main__":

    # test the dataset format and get the statistics
    train_dicts = get_train_dicts()
    test_dicts = get_test_dicts()

    print("training size:", len(train_dicts))
    print(train_dicts[1])

    with open(os.path.join("data", "preprocessed_data.pkl"), "wb") as f:
        pickle.dump(train_dicts, f)

    print("testing size:", len(test_dicts))
    print(test_dicts[3])
