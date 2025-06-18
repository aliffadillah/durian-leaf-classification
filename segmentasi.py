import cv2
import os

def segment_leaf(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (25, 40, 40)
    upper = (95, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def segment_all_images(dataset_dir, output_dir):
    for label in os.listdir(dataset_dir):
        input_path = os.path.join(dataset_dir, label)
        output_path = os.path.join(output_dir, label)
        os.makedirs(output_path, exist_ok=True)
        for file in os.listdir(input_path):
            img_path = os.path.join(input_path, file)
            segmented = segment_leaf(img_path)
            cv2.imwrite(os.path.join(output_path, file), segmented)

if __name__ == "__main__":
    segment_all_images("dataset", "segmented_dataset")
