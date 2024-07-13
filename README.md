# face_recognition
For training was use the follow dataset [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

 - [Face Alignment checkpoint](https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/blob/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat). (Used [dllib](http://dlib.net/))
 - [Base model EfficientNet_V2_S](https://github.com/pytorch/vision/blob/8f9d810a26f1e3be97e8ec48a214967accdb9016/torchvision/models/efficientnet.py#L655)  
   - [Checkpount trained on ImageNet-1K](https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth)
  
  
Project structure:  
 - ./model - you can take all checkpoints from [here](https://drive.google.com/drive/folders/1ySyiAungGjzl_AASluBQHEh-meVGmfiC?usp=drive_link)

 - [jupyter/train_baseline.ipynb](jupyter/train_baseline.ipynb) - baseline face recognition solution, no tricks(no scheduler, no augmentation, not all dataset, only 10 epoch) - test accuracy 78%
 - [face_alignment.py](face_alignment.py) - face alignment

- [jupyter/test_identification_rate_metric.ipynb](jupyter/test_identification_rate_metric.ipynb) - tests for Identificaton rate metric (TPR@FPR).
  -  [jupyter/train_triplet_margin_loss.ipynb](jupyter/train_triplet_margin_loss.ipynb) - checked IR metric on triplet loss function   
- [jupyter/train_arcface.ipynb](jupyter/train_arcface.ipynb) -  trained arcface, was use [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

- [jupyter/train_triplet_loss.ipynb](jupyter/train_triplet_loss.ipynb) -  trained triplet loss, was use [open-metric-learning](https://github.com/OML-Team/open-metric-learning)

 - [app/](app/) - [here](https://drive.google.com/file/d/1wIspqpD5LsE3LPgQtcn-WXL_6vEtA5Cx/view?usp=drive_link) is simple demo of face recognition system
   - If you want run app locally, you can just change workdir to `cd ./app` and use docker
     - docker compose build  
     - docker compose up
 
 - [dataset.py](dataset.py) - some helpers for CelebA dataset