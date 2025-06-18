import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
    return [contrast, correlation, energy, homogeneity]

def extract_from_folder(folder):
    features, labels = [], []
    for label in os.listdir(folder):
        path = os.path.join(folder, label)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            feat = extract_glcm_features(img)
            features.append(feat)
            labels.append(label)
    df = pd.DataFrame(features, columns=['contrast', 'correlation', 'energy', 'homogeneity'])
    df['label'] = labels
    return df

if __name__ == "__main__":
    df = extract_from_folder("segmented_dataset")
    df.to_csv("glcm_features.csv", index=False)
