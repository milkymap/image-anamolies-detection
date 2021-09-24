import cv2 
import numpy as np 
import torch as th

from torchvision import transforms as T 

from os import path
from glob import glob 

def th2cv(th_image):
    red, green, blue = th_image.numpy()
    return cv2.merge((blue, green, red))

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def read_image(path2image, size=None):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    if size is not None:
        return cv2.resize(cv_image, size, interpolation=cv2.INTER_CUBIC)
    return cv_image

def save_image(cv_image, path2location):
    cv2.imwrite(path2location, cv_image)

def pull_files(target_location, extension='*'):
    return glob(path.join(target_location, extension))

def prepare_image(th_image):
    normalied_th_image = th_image / th.max(th_image)
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(normalied_th_image)

def sift(image, extractor):
    keypoints, descriptor = extractor.detectAndCompute(image, None) 
    return keypoints, descriptor

def sift2mbrsift(descriptor):
    sink = []
    for row in descriptor:
        breaked_row = np.split(row, 4)
        breaked_row.reverse()
        chunks_acc = []
        for groups in breaked_row:
            breaked_chunk = np.split(groups, 4)
            for chunk in breaked_chunk:
                head, *remainder = chunk
                new_chunk = np.hstack([head, np.flip(remainder)])
                chunks_acc.append(new_chunk)
            # end loop chunk ...!
        # end loop group ...!
        sink.append(row)
        sink.append(np.hstack(chunks_acc))
    # end loop row ...!
    return np.vstack(sink) 

def compare_descriptors(source_des, target_des, threshold):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(source_des, target_des, k=2)

    valid_matches = 0 
    for left, right in matches:
        if left.distance < threshold * right.distance:
            valid_matches = valid_matches + 1 
    return valid_matches

def search_duplication(source, target_paths, size, score, threshold):
    sift_obj = cv2.SIFT_create()
    source_img = read_image(source, size)
    source_keypoints, source_features = sift(source_img, sift_obj)
    extended_source_features = sift2mbrsift(source_features)

    accumulator = []
    for current_path in target_paths:
        target_img = read_image(current_path, size)
        target_keypoints, target_features = sift(target_img, sift_obj)
        extended_target_features = sift2mbrsift(target_features)
        
        metrics = compare_descriptors(extended_source_features, extended_target_features, threshold)
        weights = metrics / np.maximum(len(source_keypoints), len(target_keypoints))
        normalized_weights = weights / 2.0

        print(weights, normalized_weights)
        
        if normalized_weights > score:
            accumulator.append({'img_name': path.split(current_path)[-1], 'score': normalized_weights})
    # end loop over other images 

    return accumulator

def build_response(response):
    return {
        'status': int(len(response) > 0),
        'contents': response   
    }






