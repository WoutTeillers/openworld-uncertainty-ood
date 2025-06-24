import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh
import json
from nltk.corpus import wordnet as wn
import os
import matplotlib.pyplot as plt
import regex as re


class DataLoader:
    def __init__(self):
        pass

    def lvis(self, max_samples=100):
        '''
            Get the LVIS dataset, with number of unique classes
            Args:
                max_samples (Int): The number of samples taken from the dataset
            Returns:
                dataset
                max_classes (Int): the number of unique classes in the dataset
        '''
        def get_number_classes(dataset):
            max_classes = -10000
            for sample in dataset:
                sample_classes = set()
                for detection in sample.detections.detections:
                    sample_classes.add(detection.label)
                
                if len(sample_classes) > max_classes:
                    max_classes = len(sample_classes)
            return max_classes
        print('loading lvis dataset')
        # Load the dataset
        dataset = fouh.load_from_hub("Voxel51/LVIS", 
                                    split="validation",
                                    max_samples=max_samples, 
                                    overwrite=True, 
                                    label_types=['detections'])
        return dataset, get_number_classes(dataset)
    
    def japan(self):
        def get_number_classes(dataset):
            max_classes = -10000
            for sample in dataset:
                sample_classes = set()
                for detection in sample.detections.detections:
                    sample_classes.add(detection.label)
                
                if len(sample_classes) > max_classes:
                    max_classes = len(sample_classes)
            return max_classes
        basepath = r"C:\Users\Wout Teillers\Documents\AI\Bachelor's project\Code\Bachelors-project2\data"
        dataset = fo.Dataset.from_dir(os.path.join(basepath, "japan_dataset"),fo.types.COCODetectionDataset)
        return dataset, get_number_classes(dataset)
    
    def open_images(self, max_samples=100):
        '''
            Get the open_images dataset, with number of unique classes
            Args:
                max_samples (Int): The number of samples taken from the dataset
            Returns:
                dataset
                max_classes (Int): the number of unique classes in the dataset
        '''
        def get_number_classes(dataset):
            max_classes = -10000
            for sample in dataset:
                sample_classes = set()
                if not sample.ground_truth:
                    continue
                for detection in sample.ground_truth.detections:
                    sample_classes.add(detection.label)
                
                if len(sample_classes) > max_classes:
                    max_classes = len(sample_classes)
            return max_classes
        
        dataset = fo.zoo.load_zoo_dataset(
              "open-images-v7",
              split="validation",
              label_types=["detections"],
              max_samples=max_samples,
          )
        
        return dataset, get_number_classes(dataset)
    
    def get_class_names(self):
        '''
        Get list of classes in the LVIS dataset and openimages dataset

        Returns:
            LVIS_classes (list[String]): list of unique classes in LVIS
            open_images_classes (list[String]): list of unique classes in open images dataset
        '''
        with open('lvis_classes.json', 'r') as file:
            LVIS_classes = json.load(file)

        with open('Openimages_classes.json', 'r') as file:
            open_images_classes = json.load(file)

        with open('japan_classes.json', 'r') as file:
            japan_classes = json.load(file)

        return LVIS_classes, open_images_classes, japan_classes
    
    def filter_class_names(self, li1, li2):
        '''
        filter two lists of classes to get a list of missing classes
        Args:
            li1 (list[String]): list of classes to keep while filtering the classes that are in li2
            li2 (list[String]): list of classes to remove
        Returns
            missing_classes (list[String]): list of missing classes
        '''
        def strip_non_alphabetic(input_string):
            # Using regular expressions to replace non-alphabetic characters with an empty string and remove everything in parentheses
            cleaned = re.sub(r'\(.*?\)', '', input_string)
            return re.sub(r'[^a-zA-Z]', '', cleaned)
        
        # Function to check if a class is in the reference list or its synonyms
        def is_in_reference_list(class_name, reference_list):
            # Get the WordNet synsets for the object class
            class_name_stripped = strip_non_alphabetic(class_name).lower()
            synsets = wn.synsets(class_name_stripped)
            for synset in synsets:
                # Check if any lemma of the synset matches any of the reference list items
                for lemma in synset.lemmas():
                    # print(strip_non_alphabetic(lemma.name()).lower())
                    if strip_non_alphabetic(lemma.name()).lower() in reference_list:
                        return True
            
            return False

        li2 = [strip_non_alphabetic(x).lower() for x in li2]
        # Create a list of object classes not in the reference list
        missing_classes = [
            obj.lower() for obj in li1 if not is_in_reference_list(obj, li2)
        ]
        
        return missing_classes
    
    def get_lvis_sample_size_and_classes(self, sample):
        '''
        Get number of unique classes in sample along with the list of unique classes
        Args:
            sample
        Returns:
            sample_size (Int): number of unique samples
            sample_classes (list[String]): list of unique classes
        '''
        sample_classes = set()
        if not sample.detections:
            return 0, []
        
        for detection in sample.detections.detections:
            sample_classes.add(detection.label)
        return len(sample_classes), list(sample_classes)
    
    def get_open_images_size_and_classes(self, sample):
        sample_classes = set()
        if not sample.ground_truth:
            return 0, []
        
        for detection in sample.ground_truth.detections:
            sample_classes.add(detection.label)
        return len(sample_classes), list(sample_classes)

    def show_data(self, dataset):
        session = fo.launch_app(dataset)

    def save_dataset(self, dataset, dir_name, label):
        export_dir = os.path.join('pred_per_label', dir_name)

        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.COCODetectionDataset,
            export_media=True,
            label_field = label
        )
    
    def merge_datasets(self, dataset1, dataset2, label1, label2):

        i = 0
        for sample, sample2 in zip(dataset1,dataset2):
            # print(f"sample {i}...........")
            i+=1
            
            filename1 = os.path.basename(sample.filepath)
            filename2 = os.path.basename(sample2.filepath)
            if filename1 == filename2:
                try:
                    if label1 in sample2 and sample2[label1]:
                        sample[label2] = sample2[label1]
                        sample.save()
                except:
                    print('no detection label')
            else:
                print('unexpected wrong file')

            

    def eval_dataset(self, dataset, label, ground_truth_label):

        # imported_dataset = fo.Dataset.from_dir(
        #     dataset_dir = os.path.join('data', dir_name),
        #     dataset_type = fo.types.COCODetectionDataset
        # )
        # dataset.rename_sample_field("detections", "new_prediction")
        # Inspect a sample to see which field contains the detections
        # sample = dataset.first()
        # print(sample)
        # Evaluate detections
        li_evaluation = dataset.list_evaluations()
        for eval in li_evaluation:
            dataset.delete_evaluation(eval)

        eval_results = dataset.evaluate_detections(
            label,   # prediction field
            gt_field=ground_truth_label,  # ground truth field
            eval_key="eval",
            compute_mAP=False,
        )

        # Print a detailed evaluation report
        eval_results.print_report()
        # Optionally, access the metrics dictionary directly:
        metrics = eval_results.metrics()
        print("Evaluation Metrics:", metrics)