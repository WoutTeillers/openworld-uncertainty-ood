from ultralytics import YOLOWorld
import fiftyone as fo


class YoloWorldLoader:
    def __init__(self, model_id="yolov8l-world.pt"):
        self.model = YOLOWorld(model_id)

    def get_sample_size(sample):
        '''
        Args:
            sample: fiftyone sample
        Returns:
            Int: number of unique samples in dataset
            list[sample]: list of samples
        '''
        sample_classes = set()
        for detection in sample.detections.detections:
            sample_classes.add(detection.label)
        return len(sample_classes), list(sample_classes)

    def predict_classes(self, sample, classes, prediction_label):
        '''
        Args:
            sample: fiftyone dataset sample
            classes: list[String] list of class names to be predicted
            prediction_label (String): label where detections can be written in fiftyone dataset
        '''
        self.model.set_classes(classes)
        confidence_scores_prediction = []
        # Run the model on the image; note that many Ultralytics YOLO models return a list
        
        results = self.model.predict(sample.filepath, conf=0.02)
        result = results[0] if isinstance(results, list) else results
        # Extract the boxes information (bounding boxes, confidence scores, class indices)
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

        # Get original image dimensions (FiftyOne requires relative coordinates)
        img = result.orig_img
        img_h, img_w = img.shape[:2]

        detections_list = []
        for box, conf, cl in zip(xyxy, confs, cls):
            # Unpack absolute coordinates
            x1, y1, x2, y2 = box

            # Convert to relative coordinates: [x, y, width, height]
            x_rel = x1 / img_w
            y_rel = y1 / img_h
            width_rel = (x2 - x1) / img_w
            height_rel = (y2 - y1) / img_h

            # Map numeric class to string label using the result.names dictionary
            label = result.names[int(cl)]
            confidence_scores_prediction.append(conf)
            detections_list.append(
                fo.Detection(
                    label=label,
                    bounding_box=[x_rel, y_rel, width_rel, height_rel],
                    confidence=conf
                )
            )

        # Create a FiftyOne Detections object
        detections = fo.Detections(detections=detections_list)

        # Add the predictions to the sample (under the "predictions" field)
        sample[prediction_label] = detections
        sample.save()
        return confidence_scores_prediction


