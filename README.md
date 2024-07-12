# face_recognition
For training was use the follow dataset:
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Put to the models folder:
 - [Face Alignment checkpoint](https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/blob/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat). (Used [dllib](http://dlib.net/))
 - [Base model EfficientNet_V2_S](https://github.com/pytorch/vision/blob/8f9d810a26f1e3be97e8ec48a214967accdb9016/torchvision/models/efficientnet.py#L655)  
   - [Checkpount trained on ImageNet-1K](https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth)
  
  
Project structure:  
 - [jupyter/train_baseline.ipynb](jupyter/train_baseline.ipynb) - baseline face recognition solution, no tricks(no scheduler, no augmentation, not all dataset, only 10 epoch) - test accuracy 78%
 -  [jupyter/train_triplet_margin_loss.ipynb](jupyter/train_triplet_margin_loss.ipynb) - implemented metric for triplet loss function, but after a little research was found cool repo [open-metric-learning](https://github.com/OML-Team/open-metric-learning), and I decided to use it for further training.
 - ./model - you can take checkpoints from [here](https://drive.google.com/drive/folders/1U6ghyTxqUuF3XHrJM5EZJ2SO)
 - /app - [here](https://drive.google.com/file/d/1wIspqpD5LsE3LPgQtcn-WXL_6vEtA5Cx/view?usp=drive_link) is simple demo of facial recognition system 