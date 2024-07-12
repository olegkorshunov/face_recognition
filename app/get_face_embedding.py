import albumentations as A
import dlib
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torchvision.models import EfficientNet, efficientnet_v2_s

from config import CFG


class GetFaceEmbedding:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.sp_predictor68 = dlib.shape_predictor(CFG.predictor_path)
        self.model = self.load_checkpoint()
        self.transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def load_checkpoint() -> EfficientNet:
        # TODO: add config for different models
        model = efficientnet_v2_s()
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=1280, out_features=2024))
        model.classifier = nn.Sequential()
        model.load_state_dict(torch.load(CFG.checkpoint_path, map_location=torch.device("cpu")))
        model.eval()
        return model

    def face_alignment(self, img, crop_size=384):
        """
        https://github.com/davisking/dlib/blob/master/python_examples/face_alignment.py
        """

        dets = self.detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            return

        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(self.sp_predictor68(img, detection))

        images = dlib.get_face_chips(img, faces, size=crop_size)
        for image in images:
            # return 1st face
            return image

        raise  # TODO: face not found

    def __call__(self, img: np.ndarray):
        crop_img = self.face_alignment(img)
        input = (self.transform(image=crop_img)["image"]).unsqueeze(0)
        with torch.no_grad():
            return self.model(input).cpu().detach().numpy()
