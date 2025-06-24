import fiftyone as fo
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class Dino:
    def __init__(self, model_id = "IDEA-Research/grounding-dino-base"):
        self.device = "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        
    def get_sample_size(self, sample):
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

    def predict_classes(self, sample, classes, prediction_label, conf_threshold=0.02):
        '''
        Args:
            sample: fiftyone dataset sample
            classes: list[String] list of class names to be predicted
            prediction_label (String): label where detections can be written in fiftyone dataset
        '''
        
        confidence_scores_prediction = []

        image = Image.open(sample.filepath)
        if image.mode == 'L':
            image = image.convert('RGB')

        classes_string = '. '.join(classes).lower() + '.'

        inputs = self.processor(images=image, text=classes_string, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)


        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=conf_threshold,
            text_threshold=conf_threshold,
            target_sizes=[(image.height, image.width)]
            )
        result = results[0]

        # Get original image dimensions (FiftyOne requires relative coordinates)
        img_w, img_h = image.size
        detections_list = []

        for box, score, text_label in zip(result["boxes"], result["scores"], result["text_labels"]):
            # if not text_label in classes[0]:
            #     continue
            box = [round(x, 2) for x in box.tolist()]
            x1, y1, x2, y2 = box
            

            # Convert to relative coordinates: [x, y, width, height]
            x_rel = x1 / img_w
            y_rel = y1 / img_h
            width_rel = (x2 - x1) / img_w
            height_rel = (y2 - y1) / img_h

            # Map numeric class to string label using the result.names dictionary
            confidence_scores_prediction.append(score.item())
            detections_list.append(
                fo.Detection(
                    label=text_label,
                    bounding_box=[x_rel, y_rel, width_rel, height_rel],
                    confidence=np.float32(score.item())
                )
            )
            # print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")


        # Create a FiftyOne Detections object
        detections = fo.Detections(detections=detections_list)

        # Add the predictions to the sample (under the "predictions" field)
        sample[prediction_label] = detections
        sample.save()
        return confidence_scores_prediction
