import argparse, json, os
from pathlib import Path
import json
import cv2, numpy as np, torch
from tqdm import tqdm
from pycocotools import mask as mask_utils
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torchvision.ops import nms


def build_predictor(weights: str, score_thr: float):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


def to_uint8_bgr(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    return img


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_inference(predictors, test_root: Path, out_json: Path, iou_thresh: float = 0.5):
    with open('test_image_name_to_ids.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    filename_to_id = {item["file_name"]: item["id"] for item in data}
    results = []

    for img_path in tqdm(sorted(test_root.glob("*.tif")), desc="Inferencing"):
        img = to_uint8_bgr(cv2.imread(str(img_path), cv2.IMREAD_COLOR))
        H, W = img.shape[:2]

        all_boxes, all_scores, all_classes, all_masks = [], [], [], []

        for predictor in predictors:
            out = predictor(img)["instances"].to("cpu")
            if len(out) == 0:
                continue
            all_boxes.append(out.pred_boxes.tensor)
            all_scores.append(out.scores)
            all_classes.append(out.pred_classes)
            if out.has("pred_masks"):
                all_masks.append(out.pred_masks)

        if not all_boxes:
            continue

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        classes = torch.cat(all_classes, dim=0)
        masks = torch.cat(all_masks, dim=0) if all_masks else None

        # Apply NMS per class
        for cls in torch.unique(classes):
            inds = (classes == cls)
            cls_boxes = boxes[inds]
            cls_scores = scores[inds]
            cls_masks = masks[inds] if masks is not None else None

            keep = nms(cls_boxes, cls_scores, iou_thresh)
            for idx in keep:
                box = cls_boxes[idx].tolist()
                score = cls_scores[idx].item()
                class_id = cls.item() + 1  # 1-based
                result = {
                    "image_id": filename_to_id[img_path.stem + ".tif"],
                    "bbox": box,
                    "score": score,
                    "category_id": class_id,
                }
                if cls_masks is not None:
                    mask = cls_masks[idx].numpy()
                    result["segmentation"] = {
                        "size": [H, W],
                        **encode_mask(mask)
                    }
                results.append(result)

    json.dump(results, open(out_json, "w"))
    print(f"Saved {len(results)} objects to {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True,
                    help="List of model weight paths for ensemble")
    ap.add_argument("--test_dir", default="data/test_release")
    ap.add_argument("--out", default="test-results.json")
    ap.add_argument("--score_thr", type=float, default=0.001)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    args = ap.parse_args()

    predictors = [build_predictor(w, args.score_thr) for w in args.weights]
    run_inference(predictors, Path(args.test_dir), Path(args.out), args.nms_iou)


if __name__ == "__main__":
    main()
