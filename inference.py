#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script ‑‑ fixed for empty‑prediction issue.
* 降低 score threshold
* 確保 NUM_CLASSES=4
* 將單通道 / 16‑bit tiff 轉成 uint8 BGR
"""

import argparse, json, os
from pathlib import Path
import json

import cv2, numpy as np, torch
from tqdm import tqdm
from pycocotools import mask as mask_utils
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def build_predictor(weights: str, score_thr: float):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


def to_uint8_bgr(img):
    """將單通道 / 16‑bit 影像轉 8‑bit BGR"""
    if img.ndim == 2:                      # 灰階 ➜ BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    return img


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_inference(predictor, test_root: Path, out_json: Path):

    # 讀取 JSON 檔案
    with open('test_image_name_to_ids.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立 filename -> id 的對應字典
    filename_to_id = {item["file_name"]: item["id"] for item in data}

    results = []

    for img_path in tqdm(sorted(test_root.glob("*.tif")), desc="Inferencing"):

        # --- 影像讀取 & 修正 ---
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = to_uint8_bgr(img)
        H, W = img.shape[:2]

        # --- 推論 ---
        out = predictor(img)["instances"].to("cpu")

        boxes   = out.pred_boxes.tensor.numpy()
        scores  = out.scores.numpy()
        classes = out.pred_classes.numpy()
        masks   = out.pred_masks.numpy() if out.has("pred_masks") else None

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            res = {
                "image_id": filename_to_id[img_path.stem + ".tif"],
                "bbox": [x1, y1, x2, y2],
                "score": float(scores[i]),
                "category_id": int(classes[i]) + 1,   # 1‑based
            }
            if masks is not None:
                res["segmentation"] = {
                    "size": [H, W],
                    **encode_mask(masks[i]),
                }
            results.append(res)

    json.dump(results, open(out_json, "w"))
    print(f"Saved {len(results)} objects to {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="./output_maskrcnn/model_final.pth")
    ap.add_argument("--test_dir", default="data/test_release")
    ap.add_argument("--out", default="test-results.json")
    ap.add_argument("--score_thr", type=float, default=0.001,
                    help="low threshold to avoid empty predictions")
    args = ap.parse_args()

    assert Path(args.weights).is_file(), f"weight file not found: {args.weights}"
    predictor = build_predictor(args.weights, args.score_thr)
    run_inference(predictor, Path(args.test_dir), Path(args.out))


if __name__ == "__main__":
    main()
