import cv2
import numpy as np
from .trt_loader import TrtModel



class Yolov7Detector():
    def __init__(self, model_path):
        self.rec_model = TrtModel(model_path)
        self.input_shape = None
        self.max_batch_size = 1
        self.stream = None
        self.input_ptr = None
        # Initialize model
        self.prepare()
        self.model_input_size = (self.rec_model.input_shapes[0][3], self.rec_model.input_shapes[0][2])
        self.latest_box = None
    
    def prepare(self):
        self.rec_model.build()
        print("self.rec_model.input_shapes", self.rec_model.input_shapes)
        self.input_shape = self.rec_model.input_shapes[0]
        self.max_batch_size = self.rec_model.max_batch_size
        self.stream = self.rec_model.stream
        self.input_ptr = self.rec_model.input
        if self.input_shape[0] == -1:
            self.input_shape = (self.max_batch_size,) + self.input_shape[1:]

        self.rec_model.run(np.zeros(self.input_shape, np.float32))
    
    def preprocess(self, image):
        self.src_shape = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(
            input_img, self.model_input_size)

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def vis(self, img, batch_pred):
        boxes, scores, class_ids = batch_pred
        boxes = self.rescale_boxes(boxes, img.shape[:2])
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            cls_id = str(cls_id)
            text = 'cls_id {}:{:.1f}%'.format(cls_id, score * 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(img, text, (x0, y0), font, 0.4, (255, 255, 0), thickness=1)
        return img   


    def run(self, batch_image, theshold):
        batch_preprocessed_image = None
        batch_preds = None
        for image in batch_image:
            preprocessed_image = self.preprocess(image)
            if batch_preprocessed_image is None:
                batch_preprocessed_image = preprocessed_image
            else:
                batch_preprocessed_image = np.vstack((batch_preprocessed_image, preprocessed_image))
            
        net_out = self.rec_model.run(batch_preprocessed_image)
        batch_preds = self.parse_processed_output(net_out, theshold)
        return batch_preds

    def rescale_boxes(self, boxes, origin_image_shape):
        # origin_image_shape (H, W)

        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [self.model_input_size[0], self.model_input_size[1], self.model_input_size[0], self.model_input_size[1]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([origin_image_shape[1], origin_image_shape[0],
                          origin_image_shape[1], origin_image_shape[0]])
        return boxes
    
    def parse_processed_output(self, outputs, threshold):
        # scores = np.squeeze(outputs[self.output_names.index('score')])
        predictions = outputs["batchno_classid_y1x1y2x2"]
        # predictions = outputs["batchno_classid_y1x1y2x2"]
        scores = np.squeeze(outputs["score"], axis=1)

        # Filter out object scores below threshold
        valid_scores = scores > threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]
        
        boxes = predictions[:, 2:]
        for box in boxes:
            box[0], box[1] = box[1], box[0]
            box[2], box[3] = box[3], box[2]

        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        unique_batch_indexs = list(set(predictions[:, 0]))
        
        pred_on_batch = []
        for unique_batch_index in unique_batch_indexs:
            boxes_on_image = []
            scores_on_image = []
            class_ids_on_image = []
            for idx, prediction in enumerate(predictions):
                batch_index = prediction[0]
                if unique_batch_index == batch_index:
                    boxes_on_image.append(boxes[idx])
                    scores_on_image.append(scores[idx])
                    class_ids_on_image.append(prediction[1])

            self.latest_box = boxes_on_image[0]
            pred_on_batch.append([boxes_on_image, scores_on_image, class_ids_on_image])               
        return pred_on_batch

