import base64
from openai import OpenAI
import os
import json
from PIL import Image
import numpy as np
import fiftyone as fo
import re


class Gpt:
    
    def __init__(self):
        pass

    def predict_classes(self, sample, classes, prediction_label, conf_threshold=0.02, model_name="gpt-4o-mini"):
        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Path to your image
        image_path = sample.filepath

        # # Getting the Base64 string
        # base64_image = encode_image(image_path)

        # response = client.chat.completions.create(
        #     model="",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "text",
        #                     "text": "Retrieve all the objects from the picture that are part of this list of object classes ([" + ','.join(classes) + "]) and give your confidence score in the range of 0 to 1 keeping only the prediction that have a confidence score of " + str(conf_threshold) + ". Format the output as json: {'scores': [], 'boxes': [[x_topleft, y_topleft, width, height]], 'labels': [String]}, where scores is a list of the confidence scores, boxes is a list of lists with the coordinates, width, height of the bounding boxes and labels is list of string labels, where the indices of the three lists are part of the same recognition.",
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        #                 },
        #             ],
        #         }
        #     ],
        # )
        base64_image = encode_image(image_path)

        # create OpenAI client
        client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY"),
        )
        image = Image.open(sample.filepath)
        if image.mode == 'L':
            image = image.convert('RGB')
        img_w, img_h = image.size

        # call the API with prompt "What can you see in the image?"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "developer",
                "content": [
                    {
                    "type": "text",
                    "text": ("""
                            You are an object recognition model capable of detecting and localizing objects within an image. 
                            Given an image with width = """ + str(img_w) + """ and height = """ + str(img_h) + """, you will receive a list of object classes that I want you to detect. 
                            Your task is to find all objects in the image that match these class labels and provide the following details for each object: 
                            1. The confidence score (ranging from 0 to 1) of the detection, ensuring that only objects with a confidence score greater than or equal to """ + str(conf_threshold) + """ are included. 
                            2. The bounding box for each detected object, given by the center (x_center, y_center) of the bounding box (in pixel coordinates) and the width and height of the bounding box (in pixel width). 
                            The bounding box should tightly enclose the object and should be calculated with respect to the objects aspect ratio and position. 
                            3. The class label for each object, corresponding to one of the classes in the provided list. 
                            Make sure that each object is localized as accurately as possible within the image. The origin point (0, 0) is at the top-left corner of the image. 
                            The format for your response should be a JSON string like the following: 
                            {'scores': [], 'boxes': [[x_center, y_center, width, height]], 'labels': [String]} 
                            Where: 
                            - scores: A list of confidence scores for each detection. 
                            - boxes: A list of bounding boxes for each object, where each bounding box is a list of four values: [x_center, y_center, width, height], representing the center of the bounding box and its dimensions. 
                            - labels: A list of strings, where each string is the class label for the corresponding object in the image. 
                            Please ensure that you only include detections that meet the confidence threshold and that the bounding boxes are as precise as possible, accurately matching the position of each object in the image. And that the output contains only the json without comments.
                            """)
                    }
                ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Can you retrieve the objects from this list of classes? [" + ','.join(classes).lower() +"]"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        
        antwoord = response.choices[0].message.content
        output = (sample.filepath, ','.join(classes), antwoord)
        if antwoord == None:
            return [], ()
        antwoord = antwoord.replace("\n", "")

        # remove comments
        antwoord = re.sub(r'#.*', '', antwoord)

        # Regular expression to match the JSON-like structure
        json_part = re.search(r'\{.*\}', antwoord)
        antwoord = json_part
        json_data = ''
        if json_part:
            # Load the JSON-like part to a dictionary (convert single quotes to double quotes for valid JSON)
            json_data = json.loads(json_part.group().replace("'", '"'))
            # print("Extracted JSON:", json_data)
        else:
            # print("No JSON-like structure found.")
            return [], ()

        result = json_data
        detections_list = []
        confidence_scores_prediction = []
        
        for box, score, text_label in zip(result["boxes"], result["scores"], result["labels"]):
            
            # print(text_label)
            box = [round(x, 2) for x in box]
            x1, y1, w, h = box
            x1 = x1 - (0.5*w)
            y1 = y1 - (0.5*h)

            # Convert to relative coordinates: [x, y, width, height]
            x_rel = x1 / img_w
            y_rel = y1 / img_h
            width_rel = w / img_w
            height_rel = h / img_h

            # Map numeric class to string label using the result.names dictionary
            confidence_scores_prediction.append(score)
            # print(f"{text_label}, {x_rel}, {y_rel}, {score}, {type(score)}")
            detections_list.append(
                fo.Detection(
                    label=text_label,
                    bounding_box=[x_rel, y_rel, width_rel, height_rel],
                    confidence=np.float32(score)
                )
            )

        # Create a FiftyOne Detections object
        detections = fo.Detections(detections=detections_list)

        # Add the predictions to the sample (under the "predictions" field)
        sample[prediction_label] = detections
        sample.save()
        return confidence_scores_prediction, output
                