# VDL_Homework3
## Introduction
This is the HW3 in visual deep learning. In this project, we should predict the pixel-level segmentation of the given image and then output all possible instances. I use the `Mask-RCNN` with pretrained backbone of ResNet-50 on `detectron2` which is called `new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ`. I set all parameters of the Mask RCNN to be trainbale and apply different learning rate to different layers of the model. I choose Adam as the optimizer and apply cosine annealing learning decay with warmup. The detailed hyper-parameters for training is shown as the following:

![image](https://github.com/user-attachments/assets/a3772d52-6a1d-4215-a4fe-c607a5e6e8fa)

For ensemble, I run the **5-fold** cross validation and combine the best models from different folds to predict the final result.


## Project structure
- `train.py` is the main function for training
- `utils.py` is the file describe how to transform the mask format from pixels to bitmask
- `dataset.py` describes how to create the dataset format that detectron2 accept
- `ensemble.py` is to the scripts of how to combine multiple models
- `inference.py` is to run testing result on one model
- `test_image_name_to_ids.json` is a file describe the mapping between testing images and image ids 

## How to run the code
- install the dependencies
  ```
  pip install -r requirements.txt
  ```
- Generate the dataset for detectron2 accepted -> it will store in `data/pre_processed.pkl`
  ```
  python dataset.py
  ```
- See the model size and training (You can assign seed to represent the fold number (0~4))
  ```
  python train.py --seed 1
  ```
- Inference on one model
  ```
  python inference.py --weights ***.pth
  ```
- Inference with ensembling (You can add many models) 
  ```
  python ensemble --weights *.pth **.pth ***.pth ****.pth
  ```

## Performance

The training record.

![image](https://github.com/user-attachments/assets/f4716e2f-1d53-425a-96b6-358b7aa1a9ee)


<hr>

The validation score of my method.

![image](https://github.com/user-attachments/assets/58c95972-22f4-4d6d-aa6b-6f1d9b6adb58)


<hr>

The score on different folds.
![螢幕擷取畫面 2025-05-07 222358](https://github.com/user-attachments/assets/1f71e59b-03d2-4ef0-83d1-27a11c4ffed6)


<hr>

The final score on public leaderboard with ensemble is **0.407** and the one on private leaderboard is **0.436**. This also indicates the generalizability of my ensemble method!


## Reference
[1] (He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask r-cnn. In Proceedings of the 
IEEE international conference on computer vision (pp. 2961-2969).)[https://arxiv.org/abs/1703.06870] 

[2] (Burel, G., & Carel, D. (1994). Detection and localization of faces on digital images. Pattern 
Recognition Letters, 15(10), 963-967.)[https://www.sciencedirect.com/science/article/abs/pii/0167865594900272] 

[3] [Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, & Ross Girshick. (2019). 
Detectron2.](https://github.com/facebookresearch/detectron2) 
