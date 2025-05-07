import os
import random
import copy
import pickle
import torch
import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
from detectron2.data import transforms as T
from detectron2.data import detection_utils, build_detection_train_loader

from dataset import get_data_by_indices


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        # 1. 將不同參數拆成多個 group
        params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # backbone 的參數用 0.1 * BASE_LR，其它用 1.0 * BASE_LR
            lr = cfg.SOLVER.BASE_LR * (0.1 if "backbone" in name else 1.0)
            params.append({
                "params": [param],
                "lr": lr,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            })

        optimizer = torch.optim.Adam(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        return optimizer
    

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logger()

    # 1. split index for train and valid
    #### load the data and get the dataset size
    with open(os.path.join("data", 'preprocessed_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    indices = [i for i in range(len(data))]
    ### shuffle the indices
    random.seed(313551058)
    random.shuffle(indices)

    fold = args.seed
    split_indices = {
        # use seed to simulate the kfold
        "train": indices[:int(len(data)*0.2*fold)] + indices[int(len(data)*0.2*(fold+1)):],
        "val":   indices[int(len(data)*0.2*fold):int(len(data)*0.2*(fold+1))],
    } 

    # 2. register the dataset
    for d, ind in split_indices.items():
        name = f"myseg_{d}"
        DatasetCatalog.register(name, lambda ind=ind, data=data: get_data_by_indices(data, ind))
        MetadataCatalog.get(name).set(thing_classes=["class1", "class2", "class3", "class4"])

    # 3. set up config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("myseg_train",)
    cfg.DATASETS.TEST = ("myseg_val",)

    cfg.DATALOADER.NUM_WORKERS = 16
    # set number of classes to 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    # load the model pretrained weights
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ.yaml")
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.TEST.EVAL_PERIOD = 250
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # change the scheduler
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.BASE_LR          = 1e-3 
    cfg.SOLVER.BASE_LR_END      = 1e-7
    cfg.SOLVER.IMS_PER_BATCH    = 3
    cfg.SOLVER.MAX_ITER         = 5000
    cfg.SOLVER.WARMUP_METHOD    = "linear"
    cfg.SOLVER.WARMUP_ITERS     = 500
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD  

    cfg.OUTPUT_DIR = "./output_maskrcnn_better_model_" + str(args.seed)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 4. training the model
    trainer = MyTrainer(cfg)

    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1000000:.2f}M")
    print(f"Trainable params: {trainable_params/1000000:.2f}M")

    cfg_file = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(cfg_file, "w") as f:
        f.write(cfg.dump())

    print(f"Config saved to {cfg_file}")

    trainer.resume_or_load(resume=False)
    trainer.train()