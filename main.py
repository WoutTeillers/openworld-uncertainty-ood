from src.yoloworld import YoloWorldLoader
from src.dino import Dino
from src.gpt import Gpt
from src.dataloader import DataLoader
from src.result_shower import ResultShower
import fiftyone as fo
import json
import os
import random


def get_dataset_and_classes(dataloader, name_data, max_samples):
    lvis_classes, openimages_classes, japan_classes = dataloader.get_class_names()

    if name_data == 'lvis':
        dataset, num_classes = dataloader.lvis(max_samples=max_samples)
        missing_classes = dataloader.filter_class_names(openimages_classes, lvis_classes)
    elif name_data == 'open_images':
        dataset,num_classes = dataloader.open_images(max_samples=max_samples)
        missing_classes = dataloader.filter_class_names(lvis_classes, openimages_classes)
    elif name_data == 'japan':
        dataset, num_classes = dataloader.japan()
        missing_classes = dataloader.filter_class_names(lvis_classes, japan_classes)

    return dataset, [x.lower() for x in missing_classes]

def get_sample_classes(dataloader, sample, name_data):
    if name_data == "lvis":
        idx, sample_classes = dataloader.get_lvis_sample_size_and_classes(sample)
    elif name_data == 'open_images':
        idx, sample_classes = dataloader.get_open_images_size_and_classes(sample)
    elif name_data == 'japan':
        idx, sample_classes = dataloader.get_lvis_sample_size_and_classes(sample)
    return idx, [x.lower() for x in sample_classes]
    

def run_yolo(name_data, prediction_label = "new_prediction", max_samples = 1000):
    yoloworldLoader = YoloWorldLoader(model_id="yolov8l-world.pt")
    
    dataloader = DataLoader()
    dataset, missing_classes = get_dataset_and_classes(dataloader, name_data, max_samples)
    confidence_scores = []
    confidence_scores1 = []
    output = []
    for i, sample in enumerate(dataset):
        
        print(f"sample {i}....................................................")
        idx, sample_classes = get_sample_classes(dataloader, sample, name_data)
        if idx == 0:
            continue
        random.seed(i)
        output.append((sample.filename, random.sample(missing_classes, idx)))
        try:
            confidence_scores.extend(yoloworldLoader.predict_classes(sample, sample_classes, prediction_label))
        except:
            print('something went wrong with predicting existing classes')

        try:
            confidence_scores1.extend(yoloworldLoader.predict_classes(sample, random.sample(missing_classes, idx), "missing_predictions"))
        except:
            print('something went wrong with predicting missing classes')
    # with open('lvis_missing_classes2.json', 'w') as file:
    #     json.dump(output, file)
    resultshower = ResultShower()
    # resultshower.plot_confidence_curve([confidence_scores, confidence_scores1], ["Predictions count existing labels", "Predictions count missing labels"], 'plots\\yolo2_' + name_data)
    dataloader.save_dataset(dataset, "yoloworld_" + name_data + "_existing_predictions", prediction_label)
    dataloader.save_dataset(dataset, "yoloworld_" + name_data + "_missing_predictions", "missing_predictions")


def run_dino(name_data, prediction_label="new_prediction", max_samples = 1000):
    dino = Dino() 
    dataloader = DataLoader()

    dataset, missing_classes = get_dataset_and_classes(dataloader, name_data, max_samples)

    confidence_scores = []
    confidence_scores1 = []
    for i, sample in enumerate(dataset):
        
        print(f"sample {i}....................................................")
        idx, sample_classes = get_sample_classes(dataloader, sample, name_data)
        if idx == 0:
            continue
        random.seed(i)

        try:
            confidence_scores.extend(dino.predict_classes(sample, sample_classes, prediction_label))
        except:
            print('Something with wrong with predicting classes')

        try:
            confidence_scores1.extend(dino.predict_classes(sample, random.sample(missing_classes, idx), "missing_predictions"))
        except:
            print('Something with wrong with predicting classes')

         
    resultshower = ResultShower()
    # resultshower.plot_confidence_curve([confidence_scores, confidence_scores1], ["Predictions count existing labels", "Predictions count missing labels"], 'plots\\dino2_' + name_data)

    dataloader.save_dataset(dataset, "dino_" + name_data + "_existing_predictions", prediction_label)
    dataloader.save_dataset(dataset, "dino_" + name_data + "_missing_predictions", "missing_predictions")



def run_GPT(name_data, prediction_label ="new_prediction", max_samples = 1000, filename = 'log_gpt.json'):
    dataloader = DataLoader()
    gptloader = Gpt()

    dataset, missing_classes = get_dataset_and_classes(dataloader, name_data, max_samples)

    log = []
    confidence_scores = []
    confidence_scores1 = []
    for i, sample in enumerate(dataset):
        
        print(f"sample {i}....................................................")
        idx, sample_classes = get_sample_classes(dataloader, sample, name_data)
        if idx == 0:
            continue
        random.seed(i)
        try:
            cur_confidence_scores, output = gptloader.predict_classes(sample, sample_classes, prediction_label)
            log.append(output)
            confidence_scores.extend(cur_confidence_scores)
        except Exception as error:
            print('Something went wrong with predicting classes, ', error)
            log.append((None,None,None))
        try:
            cur_confidence_scores, output = gptloader.predict_classes(sample, random.sample(missing_classes, idx), "missing_predictions")
            log.append(output)
            confidence_scores1.extend(cur_confidence_scores)
        except Exception as error:
            print('Something went wrong with predicting missing classes, ', error)
            log.append((None,None,None))



    with open(filename, 'w') as file:
        json.dump(log, file)
    resultshower = ResultShower()
    # resultshower.plot_confidence_curve([confidence_scores, confidence_scores1], ["Predictions count existing labels", "Predictions count missing labels"], 'plots\\yolo2_' + name_data)
    dataloader.save_dataset(dataset, "gpt_" + name_data + "_existing_predictions", prediction_label)
    dataloader.save_dataset(dataset, "gpt_" + name_data + "_missing_predictions", "missing_predictions")

if __name__ == '__main__':

    run_GPT('lvis', filename='20250417_log_gpt_lvis.json')
    run_GPT('open_images', filename='20250417_log_gpt_open_images.json')
    run_GPT('japan', filename='20250502_log_gpt_japan.json')

    run_yolo('lvis')
    run_yolo('open_images')
    run_yolo('japan')

    run_dino('lvis')
    run_dino('open_images')
    run_dino('japan')


        
