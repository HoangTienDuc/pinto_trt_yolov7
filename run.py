import cv2
import random
from pathlib import Path
from app.trt_yolov7 import Yolov7Detector

def simulate_pred_on_batch(predictor, image_paths):
    batch_image = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch_image.append(image)
    batch_preds = predictor.run(batch_image, theshold=0.7)
    for image_path, batch_pred in zip(image_paths, batch_preds):
        print(f"image_path: {image_path}, batch_pred {batch_pred}")

if __name__=='__main__':
    image_folder_path = "data/images"
    MODEL_PATH = "models/sgie_81k.trt"
    predictor = Yolov7Detector(MODEL_PATH)
    
    image_folder = Path(image_folder_path)
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpeg"))

    index = 0
    while True:
        batch_image_file = []
        print("*"* 50)
        for i in range(random.randint(1, 4)):
            index += 1
            batch_image_file.append(str(image_files[index]))
        simulate_pred_on_batch(predictor, batch_image_file)


    