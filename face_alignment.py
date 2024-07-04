"""
https://github.com/davisking/dlib/blob/master/python_examples/face_alignment.py
"""

import os

import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CFG

DETECTOR = dlib.get_frontal_face_detector()
SP_PREDICTOR68 = dlib.shape_predictor(CFG.predictor_path)


def face_alignment(face_file_path, crop_size=CFG.crop_size):
    # Load the image using Dlib
    img = dlib.load_rgb_image(face_file_path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = DETECTOR(img, CFG.upsample)

    num_faces = len(dets)
    if num_faces == 0:
        return

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(SP_PREDICTOR68(img, detection))

    # Get the aligned face images
    # Optionally:
    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    images = dlib.get_face_chips(img, faces, size=crop_size)
    for image in images:
        # return 1st face
        return image


def process_face_alignment(img_folder_src, img_folder_dst, crop_size=CFG.crop_size):
    assert os.path.exists(img_folder_src), f"Path not fouund: {img_folder_src=}"
    assert not os.path.exists(img_folder_dst), f"Folder alredy exist: {img_folder_src=}"
    if not os.path.exists(img_folder_dst):
        os.mkdir(img_folder_dst)
    not_processed_imgs = 0
    image_list = sorted(os.listdir(img_folder_src))
    t = tqdm(image_list)
    for img_name in t:
        original_image_path = os.path.join(img_folder_src, img_name)
        algn_crop_img = face_alignment(original_image_path, crop_size)
        if isinstance(algn_crop_img, np.ndarray):
            cv2.imwrite(os.path.join(img_folder_dst, img_name), cv2.cvtColor(algn_crop_img, cv2.COLOR_RGB2BGR))
        else:
            not_processed_imgs += 1
        t.set_description(f"Not found faces: {not_processed_imgs}")


if __name__ == "__main__":
    # TODO: add multiprocess
    process_face_alignment(CFG.img_folder_src, CFG.img_folder_dst)
