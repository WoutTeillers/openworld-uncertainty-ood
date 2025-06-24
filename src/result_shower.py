import matplotlib.pyplot as plt
import fiftyone as fo
import os
import matplotlib.pyplot as plt
import fiftyone.utils.huggingface as fouh
from src.dataloader import DataLoader
from fiftyone import ViewField as F
import numpy as np
import seaborn as sns
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import rc
import scipy.stats
from sklearn.preprocessing import LabelEncoder

# TODO make latex work in matplotlib
plt.rcParams['text.usetex'] = True
rc('font', family='serif', serif='Times New Roman')

class ResultShower:

    def __init__(self):
        dataset_ori_lvis, dataset_ori_openimages, dataset_ori_japan = self.load_ori_datasets()

        self.dataset_ori_lvis = dataset_ori_lvis
        self.dataset_ori_openimages = dataset_ori_openimages
        self.dataset_ori_japan = dataset_ori_japan

    def plot_confidence_curve(self, li_confidence_scores, li_labels, name):
        '''
        Plot the confidence curve of the predictions of the model
        '''

        # Create a list of thresholds from 0 to 1
        thresholds = [i / 100 for i in range(101)]  # From 0 to 1, in steps of 0.01
        colors = [
            "blue",
            "green",
            "red",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "orange",
            "purple",
            "brown"
        ]
        # Count the number of predictions above each threshold
        prediction_counts = []

        for confidence_scores, label, color in zip(li_confidence_scores, li_labels, colors):
            prediction_counts = []
            for threshold in thresholds:
                count = sum([1 for score in confidence_scores if score >= threshold])
                prediction_counts.append(count)
            # Plot the confidence curve
            plt.plot(thresholds, prediction_counts, label=label, color=color)
        # plt.plot(thresholds, prediction_counts_existing_classes, label="Predictions count existing labels", color='r')
        # plt.plot(thresholds, prediction_counts_missing_in_training, label="Predictions count missing labels but in training", color='y')
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Number of Predictions")
        plt.title("Confidence Curve")
        plt.ylim(0, 500)
        plt.grid(True)
        plt.legend()

        # plt.savefig(name + "_confidence_curve.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_confidence_distribution_outline(self, li_confidence_scores, li_labels, name):
        """
        Plot the distribution of confidence scores as an outline (line graph) for each label.
        """

        colors = [
            "blue",
            "green",
            "red",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "orange",
            "purple",
            "brown"
        ]
        
        # For each label, plot a histogram as an outline using histtype='step'
        for confidence_scores, label, color in zip(li_confidence_scores, li_labels, colors):
            plt.hist(confidence_scores, bins=20, histtype="step", label=label, color=color, linewidth=2)
        
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.title("Confidence Distribution (Outline)")
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.savefig(name + "_confidence_distribution_outline.png", dpi=300, bbox_inches="tight")
        plt.show()

    def load_ori_datasets(self):
        def make_ds_lower(ds, field):
            for sample in ds:
                if field not in sample or not sample[field]:
                    continue
                for det in sample[field].detections:
                    det.label = det.label.lower()
                sample.save()
            newset = set()
            for sample in ds:
                if field not in sample or not sample[field]:
                    continue
                for det in sample[field].detections:
                    newset.add(det.label)
            
        dataset_ori_openimages = fo.zoo.load_zoo_dataset(
              "open-images-v7",
              split="validation",
            #   overwrite=True,
              label_types=["detections"],
              max_samples=1000,
          )

        dataset_ori_lvis = fouh.load_from_hub("Voxel51/LVIS", 
                                            split="validation",
                                            max_samples=1000, 
                                            overwrite=True, 
                                            label_types=['detections'])

        dataset_ori_japan = fo.Dataset.from_dir(os.path.join('data', "japan_dataset"),fo.types.COCODetectionDataset)

        make_ds_lower(dataset_ori_lvis, 'detections')
        make_ds_lower(dataset_ori_openimages, 'ground_truth')
        make_ds_lower(dataset_ori_japan, 'detections')


        return dataset_ori_lvis, dataset_ori_openimages, dataset_ori_japan

    def load_yolo_datasets(self, dir = 'data2', 
                           dir_names = ['yoloworld_open_images_existing_predictions', 'yoloworld_open_images_missing_predictions', 
                                        'yoloworld_lvis_existing_predictions', 'yoloworld_lvis_missing_predictions', 
                                        'yoloworld_japan_existing_predictions2', 'yoloworld_japan_missing_predictions2']):
        
        '''loads and merges the raw datasets after running YOLO-World on the datasets'''
        
        dataloader = DataLoader()

        # load open images for YOLO-world
        dataset_yolo_openimages = fo.Dataset.from_dir(os.path.join(dir, dir_names[0]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[1]),
                                    dataset_type = fo.types.COCODetectionDataset)

        # load LVIS for YOLO-world


        dataloader.merge_datasets(dataset_yolo_openimages, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_yolo_openimages, self.dataset_ori_openimages, 'ground_truth', 'ground_truth')

        print('merging done for openimages')
        dataset_yolo_lvis = fo.Dataset.from_dir(os.path.join(dir, dir_names[2]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[3]),
                                    dataset_type = fo.types.COCODetectionDataset)

        
        dataloader.merge_datasets(dataset_yolo_lvis, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_yolo_lvis, self.dataset_ori_lvis, 'detections', 'ground_truth')

        print('merging done for LVIS')
        # load JUS for YOLO-World

        dataset_yolo_japan = fo.Dataset.from_dir(os.path.join(dir, dir_names[4]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[5]),
                                    dataset_type = fo.types.COCODetectionDataset)


        # dataset3 = fo.load_dataset('open-images-v7-validation-100')

        
        dataloader.merge_datasets(dataset_yolo_japan, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_yolo_japan, self.dataset_ori_japan, 'detections', 'ground_truth')
        print('merging done for JUS')

        return dataset_yolo_lvis, dataset_yolo_openimages, dataset_yolo_japan

    def load_dino_datasets(self, dir = 'data2',
                           dir_names = ['dino_open_images_existing_predictions', 'dino_open_images_missing_predictions', 
                                        'dino_lvis_existing_predictions', 'dino_lvis_missing_predictions', 
                                        'dino_japan_existing_predictions2', 'dino_japan_missing_predictions2']):
        
        '''loads and merges the raw datasets after running Grounding Dino on the datasets'''

        dataloader = DataLoader()

        # load open images for Dino

        dataset_dino_openimages = fo.Dataset.from_dir(os.path.join(dir, dir_names[0]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[1]),
                                    dataset_type = fo.types.COCODetectionDataset)

        
        dataloader.merge_datasets(dataset_dino_openimages, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_dino_openimages, self.dataset_ori_openimages, 'ground_truth', 'ground_truth')


        # load LVIS dataset for Dino

        dataset_dino_lvis = fo.Dataset.from_dir(os.path.join(dir, dir_names[2]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[3]),
                                    dataset_type = fo.types.COCODetectionDataset)


        
        dataloader.merge_datasets(dataset_dino_lvis, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_dino_lvis, self.dataset_ori_lvis, 'detections', 'ground_truth')

        # load JUS for DINO

        dataset_dino_japan = fo.Dataset.from_dir(os.path.join(dir, dir_names[4]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[5]),
                                    dataset_type = fo.types.COCODetectionDataset)

        
        dataloader.merge_datasets(dataset_dino_japan, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_dino_japan, self.dataset_ori_japan, 'detections', 'ground_truth')

        return dataset_dino_lvis, dataset_dino_openimages, dataset_dino_japan
    
    def remove_combi_predictions_dino(self, dataset_dino_lvis, dataset_dino_openimages, dataset_dino_japan):
        '''Removes all combination predictions from the dataset, to keep only the asked predictions of the prompt'''
        
        def remove_combi_predictions(ds, field, d_classes):
            
            if field == "missing_predictions":
                for sample in ds:
                    if field in sample and sample[field]:
                        sample[field].detections = [
                            det for det in sample[field].detections if det.label in d_classes[sample.filename]
                        ]
                    sample.save()
            else:
                for sample in ds:
                    if field in sample and sample[field]:
                        li_classes = list(set([det.label for det in sample.ground_truth.detections]))
                        sample[field].detections = [
                            det for det in sample[field].detections if det.label in li_classes
                        ]
                    sample.save()


        dataloader = DataLoader()
        lvis_classes, openimages_classes, japan_classes = dataloader.get_class_names()

        with open('lvis_missing_classes.json', 'r') as file:
            missing_classes_lvis = json.load(file)

        with open('open_images_missing_classes.json', 'r') as file:
            missing_classes_OI = json.load(file)

        with open('japan_missing_classes.json', 'r') as file:
            missing_classes_JUS = json.load(file)

        d_missing_classes_lvis = {}
        for k, item in missing_classes_lvis:
            d_missing_classes_lvis[k] = item
        d_missing_classes_OI = {}
        for k, item in missing_classes_OI:
            d_missing_classes_OI[k] = item
        d_missing_classes_JUS = {}
        for k, item in missing_classes_JUS:
            d_missing_classes_JUS[k] = item

        remove_combi_predictions(dataset_dino_lvis, 'missing_predictions', d_missing_classes_lvis)
        remove_combi_predictions(dataset_dino_openimages, 'missing_predictions', d_missing_classes_OI)
        remove_combi_predictions(dataset_dino_japan, 'missing_predictions', d_missing_classes_JUS)

        remove_combi_predictions(dataset_dino_lvis, 'detections', {})
        remove_combi_predictions(dataset_dino_openimages, 'detections', {})
        remove_combi_predictions(dataset_dino_japan, 'detections', {})

    def remove_overlapping_prediction_with_nms(self, dataset_dino_lvis, dataset_dino_openimages, dataset_dino_japan):
        def compute_iou(box1, box2):
            """
            Compute the Intersection over Union (IoU) between two bounding boxes.
            Each box is represented as [x1, y1, width, height] (top-left corner, width, height).
            """
            # Convert from [x1, y1, width, height] to [x1, y1, x2, y2] (top-left, bottom-right)
            x1_1, y1_1, w1, h1 = box1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1

            x1_2, y1_2, w2, h2 = box2
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2

            # Compute the coordinates of the intersection rectangle
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)

            # Compute the area of the intersection
            intersection_area = abs(x2_inter - x1_inter) * abs(y2_inter - y1_inter)

            # Compute the area of both boxes
            box1_area = w1 * h1
            box2_area = w2 * h2

            # Compute the IoU
            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            return iou

        def non_maximum_suppression(detections, iou_threshold=0.5):
            """
            Perform Non-Maximum Suppression (NMS) to filter out overlapping detections.
            Detections should be a list of dictionaries, each containing:
                - label (str or int)
                - confidence (float)
                - bounding box (list or array: [x1, y1, width, height])
            """
            # Sort detections by confidence in descending order
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            # List to hold the final selected detections
            selected_detections = []

            # Iterate through detections and apply NMS
            for detection in detections:
                keep = True
                # Check if the current detection overlaps with any of the selected detections
                for selected in selected_detections:
                    # Only check for overlap if the label is the same
                    if detection['label'] == selected['label']:
                        iou = compute_iou(detection['bbox'], selected['bbox'])
                        if iou > iou_threshold:
                            keep = False
                            break  # Exit the loop early as we've found an overlapping detection
                if keep:
                    selected_detections.append(detection)

            return selected_detections

        def filter_overlapping_detections(ds, pred_label, iou_threshold=0.5):

            for sample in ds:
                
                if pred_label in sample and sample[pred_label]:
                    detections = sample[pred_label].detections
                        
                    # Prepare detections for NMS
                    detections_list = []
                    for detection in detections:
                        detection_data = {
                            'label': detection.label,
                            'confidence': detection.confidence,
                            'bbox': detection.bounding_box
                        }
                        detections_list.append(detection_data)

                    # Apply Non-Maximum Suppression (NMS)
                    filtered_detections = non_maximum_suppression(detections_list, iou_threshold)

                    # Create a list of Detections from filtered detections
                    new_detections = []
                    for filtered in filtered_detections:
                        new_detection = fo.Detection(
                            label=filtered['label'],
                            confidence=filtered['confidence'],
                            bounding_box=filtered['bbox']
                        )
                        new_detections.append(new_detection)

                    # Update the detections in the sample
                    sample[pred_label] = fo.Detections(detections=new_detections)
                    sample.save()

        for ds in [dataset_dino_lvis, dataset_dino_openimages, dataset_dino_japan]:
            filter_overlapping_detections(ds, 'detections')
            filter_overlapping_detections(ds, 'missing_predictions')

    def load_gpt_datasets(self, dir = 'data2',
                          dir_names = ['gpt_open_images_existing_predictions', 'gpt_open_images_missing_predictions', 
                                        'gpt_lvis_existing_predictions', 'gpt_lvis_missing_predictions', 
                                        'gpt_japan_existing_predictions2', 'gpt_japan_missing_predictions2']):
        
        '''loads and merges the raw datasets after running gpt-4o on the datasets'''


        dataloader = DataLoader()

        # load open images for GPT

        dataset_gpt_openimages = fo.Dataset.from_dir(os.path.join(dir, dir_names[0]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[1]),
                                    dataset_type = fo.types.COCODetectionDataset) 

        dataloader.merge_datasets(dataset_gpt_openimages, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_gpt_openimages, self.dataset_ori_openimages, 'ground_truth', 'ground_truth')

        # load lvis for GPT

        dataset_gpt_lvis = fo.Dataset.from_dir(os.path.join(dir, dir_names[2]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[3]),
                                    dataset_type = fo.types.COCODetectionDataset) 

        dataloader.merge_datasets(dataset_gpt_lvis, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_gpt_lvis, self.dataset_ori_lvis, 'detections', 'ground_truth')

        # load japan for GPT

        dataset_gpt_japan = fo.Dataset.from_dir(os.path.join(dir, dir_names[4]),
                                    dataset_type = fo.types.COCODetectionDataset)
        dataset2 = fo.Dataset.from_dir(os.path.join(dir, dir_names[5]),
                                    dataset_type = fo.types.COCODetectionDataset)

        
        dataloader.merge_datasets(dataset_gpt_japan, dataset2, 'detections','missing_predictions')
        dataloader.merge_datasets(dataset_gpt_japan, self.dataset_ori_japan, 'detections', 'ground_truth')

        return dataset_gpt_lvis, dataset_gpt_openimages, dataset_gpt_japan
    

    def make_combination_datasets(self, dataset1, dataset2, name):
        '''combine two datasets by extending the two'''
        existing_datasets = fo.list_datasets()
        if name in existing_datasets:
            fo.delete_dataset(name)

        dataset_combined = fo.Dataset(name=name)
        dataset_combined.add_samples(dataset1.clone())
        dataset_combined.add_samples(dataset2.clone())

        return dataset_combined
    
    
    def filter_missing_predictions(self, ds, my_dict):
        for sample in ds:
            # Check if the filename is in my_dict
            if sample.filename in my_dict:
                # Filter the detections to remove those that are in my_dict[sample.filename]
                if 'missing_predictions' in sample and sample['missing_predictions']:
                    filtered_detections = [
                        detection for detection in sample['missing_predictions'].detections
                        if detection.label not in my_dict[sample.filename]
                    ]
                    sample['missing_predictions'] = fo.Detections(detections=filtered_detections)
                    # Update the sample's missing_predictions with the filtered detections
                    
                    # Save the changes to the sample
                    sample.save()
        
        # 
    def plot_correlation(self, li_datasets, li_names):
        '''plot the correlation between size of bounding box and confidence scores'''
        print('new plot correlation')
        label = 'detections'
        fig, axes = plt.subplots(9, 2, figsize=(15, 15))  # 3 rows, 3 columns

        # Flatten the 2D axes array for easier iteration
        axes = axes.flatten()
        count = 0
        for ds, name in zip(li_datasets, li_names):
            print(name)
            for label in ['detections', 'missing_predictions']:
                # Prepare lists to store diagonals and confidence scores
                diagonals = []
                scores = []

                # Loop through the dataset and process each sample
                for sample in ds:
                    if not sample[label]:
                        continue

                    detections = sample[label].detections  # Accessing the 'detections' list
                    for detection in detections:
                        # Access the bounding box and confidence
                        bbox = detection.bounding_box  # Get the bounding box [x, y, width, height]
                        confidence = detection.confidence  # Confidence score (may be None)

                        # Check if the confidence is None; if so, set it to a default value (e.g., 0)
                        if confidence is None:
                            confidence = 0.0

                        # Calculate the diagonal of the bounding box using the Pythagorean theorem
                        width = bbox[2]  # width
                        height = bbox[3]  # height
                        diagonal = np.sqrt(width**2 + height**2)  # Diagonal using Pythagorean theorem

                        # Append the diagonal and confidence score to the lists
                        diagonals.append(diagonal)
                        scores.append(confidence)
                # Calculate the Pearson correlation coefficient
                correlation_matrix = np.corrcoef(diagonals, scores)
                correlation = correlation_matrix[0, 1]

                print("Pearson Correlation Coefficient:", correlation)

                # Plot the diagonal vs. confidence scores
                axes[count].scatter(diagonals, scores)
                axes[count].set_xlabel(r"\textbf{Diagonal}")
                axes[count].set_ylabel(r"\textbf{Confidence Score}")
                axes[count].set_title(f"{name} + {label}")
                count+=1

        plt.tight_layout()
        plt.savefig("plots/correlation.pdf", format="pdf", bbox_inches="tight")
        plt.show()


    def plot_density_confidences(self, li_datasets, li_names, num=2):
        '''Plot density histogram of datasets confidence scores for missing labels and existing labels
        with column and row titles on the outside of the grid'''

        # These would normally be part of your class or inputs
        row_titles = ["YOLO-World", "Grounding DINO", "GPT-4o"]
        col_titles = ["LVIS", "Open Images", "JUS"]

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 3 rows, 3 columns

        count = 0
        for row in range(3):
            for col in range(3):
                ax = axes[row, col]
                ds = li_datasets[count]
                name = li_names[count]

                confidence_scores = []
                confidence_scores2 = []

                for sample in ds:
                    if 'missing_predictions' in sample and sample['missing_predictions']:
                        for detection in sample['missing_predictions'].detections:
                            confidence_scores.append(detection['confidence'])
                    if 'detections' in sample and sample['detections']:
                        for detection in sample['detections'].detections:
                            confidence_scores2.append(detection['confidence'])

                bin_edges = np.arange(0, 1.1, 0.1)

                if num == 0:
                    ax.hist(confidence_scores2, bins=bin_edges, density=True, alpha=0.5,
                            label='Prediction Existing Labels', color='green', edgecolor='black')
                elif num == 1:
                    ax.hist(confidence_scores, bins=bin_edges, density=True, alpha=0.5,
                            label='Prediction Missing Labels', color='blue', edgecolor='black')
                elif num == 2:
                    ax.hist(confidence_scores, bins=bin_edges, density=True, alpha=0.5,
                            label='Prediction Missing Labels', color='blue', edgecolor='black')
                    ax.hist(confidence_scores2, bins=bin_edges, density=True, alpha=0.5,
                            label='Prediction Existing Labels', color='green', edgecolor='black')

                ax.set_xlabel(r"\textbf{Confidence Score}")
                ax.set_ylabel(r"\textbf{Density}")
                ax.set_xlim((0, 1))
                ax.legend()
                count += 1


        for ax, col in zip(axes[0], col_titles):
            ax.set_title(col)

        # Add external row titles
        for i, row_title in enumerate(row_titles):
            fig.text(0.04, 0.80 - i * 0.31, row_title, va='center', rotation='vertical', fontsize=12)
            

        plt.suptitle('Density Histogram of the Distribution of Prediction Confidence Scores')
        plt.tight_layout(rect=[0.05, 0, 1, 1])
        plt.savefig("plots/density" + str(num) +".pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def calculate_map(self, ground_truth, predictions):
        '''Calculate mAP score for a single sample (average precision).'''
        return average_precision_score(ground_truth, predictions)

    def plot_boxplot_by_classes(self, data, y):
        """
        Creates a boxplot for different values of y (number of classes).
        
        Parameters:
        - data: List or array of values to plot
        - y: List or array of the corresponding class labels (the number of classes)
        """
        # Create a DataFrame with the data and class labels
        df = pd.DataFrame({'data': data, 'y': y})
        
        # Plot the boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='y', y='data', data=df)
        
        plt.title('Boxplot by Number of Classes')
        plt.xlabel('Number of Classes')
        plt.ylabel('Data')
        plt.show()

    def plot_metrics_vs_classes(self, li_datasets, li_names, metric='AUC'):
        """
        For each model in li_names / li_datasets, collect all metric scores grouped by
        the number of distinct ground‐truth classes in each sample, and then plot a figure
        of vertical boxplots (one box per unique class‐count).
        """
        from collections import defaultdict

        encoder = LabelEncoder()

        for ds, name in zip(li_datasets, li_names):
            # Group metric values by number of classes
            grouped_scores = defaultdict(list)

            for sample in ds:
                # Skip if no ground_truth or detections
                if 'ground_truth' not in sample or not sample['ground_truth']:
                    continue
                if 'detections' not in sample or not sample['detections']:
                    continue

                # Extract ground‐truth labels
                ground_truth = [pred.label for pred in sample.ground_truth.detections]
                if not ground_truth:
                    continue

                # Extract predicted confidences and true/false labels
                predictions = sample.detections.detections
                confidences = [pred.confidence for pred in predictions]
                y_true = [1 if pred.eval == 'tp' else 0 for pred in predictions]
                labels = [pred.label for pred in predictions]
                encoded_labels = encoder.fit_transform(labels)

                # If anything is empty, skip
                if not (confidences and y_true and labels):
                    continue

                # Count unique classes in ground truth
                num_cls = len(np.unique(ground_truth))

                # Compute ECE once (always stored, even if not plotting)
                ece_score = self.expected_calibration_error(confidences, y_true)[0]

                # AUC: only if at least one positive and one negative
                if len(np.unique(y_true)) > 1:
                    auc_score = roc_auc_score(y_true, confidences)
                else:
                    # skip if only one class in y_true
                    continue

                # mAP / AP: same check
                if len(np.unique(y_true)) > 1:
                    ap_score = average_precision_score(y_true, confidences)
                else:
                    continue

                # Decide which metric to store
                if metric == 'AUC':
                    grouped_scores[num_cls].append(auc_score)
                elif metric == 'ECE':
                    grouped_scores[num_cls].append(ece_score)
                elif metric == 'mAP':
                    grouped_scores[num_cls].append(ap_score)
                else:
                    raise ValueError("Invalid metric. Choose from 'AUC', 'ECE', or 'mAP'.")

            # If no data collected, skip plotting
            if not grouped_scores:
                print(f"No valid samples found for model '{name}' when computing {metric}. Skipping.")
                continue

            # Sort the unique class counts
            sorted_class_counts = sorted(grouped_scores.keys())
            # Build a list of lists in the same sorted order
            boxplot_data = [grouped_scores[c] for c in sorted_class_counts]

            # Prepare axis labels and title
            if metric == 'AUC':
                ylabel = 'AUC Score'
                title = f'AUC vs Number of Classes for {name}'
            elif metric == 'ECE':
                ylabel = 'ECE Score'
                title = f'ECE vs Number of Classes for {name}'
            else:  # metric == 'mAP'
                ylabel = 'mAP Score'
                title = f'mAP vs Number of Classes for {name}'

            # Create a figure with one axis
            fig, ax = plt.subplots(figsize=(8, 6))
            # Draw vertical boxplots
            bp = ax.boxplot(
                boxplot_data,
                positions=sorted_class_counts,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor='lightgray', edgecolor='black'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', color='black', alpha=0.5)
            )

            ax.set_xlabel('Number of Classes in Ground Truth', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            # ax.set_title(title, fontsize=14)
            ax.set_xticks(sorted_class_counts)
            ax.set_xticklabels([str(c) for c in sorted_class_counts])
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.xlim(0,20)
            plt.ylim(-0.1,1.1)

            plt.tight_layout()
            plt.savefig(f"plots/scatter_{name}_{metric}.pdf", format="pdf", bbox_inches="tight")
            plt.show()

    def plot_density_confidences_seperate(self, li_datasets, li_names, num=2):
        '''Plot density histogram of datasets' confidence scores for missing labels and existing labels'''

        # These would normally be part of your class or inputs
        row_titles = ["YOLO-World", "Grounding DINO", "GPT-4o"]
        col_titles = ["LVIS", "Open Images", "JUS"]

        count = 0
        for row in range(3):
            for col in range(3):
                ds = li_datasets[count]
                name = li_names[count]

                confidence_scores = []
                confidence_scores2 = []

                for sample in ds:
                    if 'missing_predictions' in sample and sample['missing_predictions']:
                        for detection in sample['missing_predictions'].detections:
                            confidence_scores.append(detection['confidence'])
                    if 'detections' in sample and sample['detections']:
                        for detection in sample['detections'].detections:
                            confidence_scores2.append(detection['confidence'])

                bin_edges = np.arange(0, 1.1, 0.1)

                # Create a separate plot for each dataset/model
                plt.figure(figsize=(8, 6))  # You can adjust the size of the individual plots
                if num == 0:
                    plt.hist(confidence_scores2, bins=bin_edges, density=True, alpha=0.5,
                            label='Prediction Existing Labels', color='green', edgecolor='black')
                elif num == 1:
                    plt.hist(confidence_scores, bins=bin_edges, density=True, alpha=0.5,
                            label='Prediction Missing Labels', color='blue', edgecolor='black')
                elif num == 2:
                    plt.hist(confidence_scores, bins=bin_edges, density=True, alpha=0.5,
                            label='Prediction Missing Labels', color='blue', edgecolor='black')
                    plt.hist(confidence_scores2, bins=bin_edges, density=True, alpha=0.5,
                            label='Prediction Existing Labels', color='green', edgecolor='black')

                plt.xlabel(r"\textbf{Confidence Score}")
                plt.ylabel(r"\textbf{Density}")
                plt.xlim((0, 1))
                plt.legend()
                # plt.title(f'Density Histogram for {row_titles[row]} - {col_titles[col]}')

                # Save the plot for each dataset/model
                plt.tight_layout()
                plt.savefig(f"plots/density{num}/density_{row_titles[row]}_{col_titles[col]}_{num}.pdf", format="pdf", bbox_inches="tight")
                plt.show()  # Display the plot

                count += 1

    def give_predictions_if_none(self, ds):
        def get_length_labels(ds,label):
            li_labels = []
            for sample in ds:
                if sample[label]:
                    li_labels.extend([pred.label for pred in sample[label].detections])
            # print(set(li_labels))
            return len(set(li_labels)), li_labels

        for sample in ds:
            sample_labels = []
            try:
                if sample['ground_truth']:
                    sample_labels = list(set([det.label.lower() for det in sample.ground_truth.detections]))
                sample_label_detections = []
                if sample.detections:
                    sample_label_detections = list(set([det.label.lower() for det in sample.detections.detections]))
                missing_labels = [label for label in sample_labels if not label in sample_label_detections]
                detections_list = []

                for label in missing_labels:
                    detections_list.append(
                        fo.Detection(
                            label=label,
                            bounding_box=[0,0,0,0],
                            confidence=0
                        )
                    )

                if missing_labels:
                    if 'detections' in sample and sample.detections:
                        existing_detections = sample.detections.detections
                        existing_detections.extend(detections_list)
                        detections = fo.Detections(detections=existing_detections)
                        sample.detections = None
                        sample['detections'] = detections
                        sample.save()
                    else:
                        existing_detections = []
                        existing_detections.extend(detections_list)
                        detections = fo.Detections(detections=existing_detections)
                        sample.detections = None
                        sample['detections'] = detections
                        sample.save()
                sample.save()
            except:
            
                print('not ground_truth')

    def eval_datasets(self, li_datasets):
        '''evaluate a list of datasets, calculate accuracy, precision, recall, fscore, and mAP scores'''

        def sub_eval_dataset(dataset, label, ground_truth_label, pred_labels):
            li_evaluation = dataset.list_evaluations()
            for eval in li_evaluation:
                dataset.delete_evaluation(eval)

            eval_results = dataset.evaluate_detections(
                label,   # prediction field
                gt_field=ground_truth_label,  # ground truth field
                eval_key="eval",
                compute_mAP=True,
                classes=pred_labels,
                iou=0.5, 
            )

            # Print a detailed evaluation report
            # eval_results.print_report()
            # Optionally, access the metrics dictionary directly:
            metrics = eval_results.metrics()
            print("Evaluation Metrics:", metrics)
            print(f"mAP score = {eval_results.mAP()}")

        def get_prediction_labels(ds):
            pred_labels = []
            for sample in ds:
                try:
                    if 'ground_truth' in sample and sample['ground_truth']:
                        for det in sample.ground_truth.detections:
                            pred_labels.append(det.label)
                except:
                    print('sample has not ground truth')
            pred_labels = list(set(pred_labels))
            return pred_labels
        
        for ds in li_datasets:
            pred_labels = get_prediction_labels(ds)
            sub_eval_dataset(dataset=ds, label='detections', ground_truth_label='ground_truth', pred_labels=pred_labels)

    def check_gpt_output(self, datafiles):
        import json
        import regex as re


        def is_valid(str):
            if str == None:
                return False
            str = str.replace("\n", "")

            # Regular expression to match the JSON-like structure
            json_part = re.search(r'\{.*\}', str)
            antwoord = json_part
            json_data = ''
            
            if json_part:
                return True
            else:
                # print("No JSON-like structure found.")
                return False

        for datafile in datafiles:
            # Open the JSON file
            with open(datafile, 'r') as file:
                # Load the data from the file
                data = json.load(file)

            # Now you can work with the data as a Python dictionary

            count = 0
            for i, row in enumerate(data):
                if row:
                    if is_valid(row[2]):
                        count+=1

            print(f"{count/len(data)*100} % good responses in {datafile}")



    def plot_number_predictions(self, li_models, li_names):
        '''Plot a single grouped boxplot of existing vs missing predictions per sample for each model'''

        records = []

        for ds, name in zip(li_models, li_names):
            for sample in ds:
                existing = len(sample.detections.detections) if 'detections' in sample and sample.detections else 0
                missing = len(sample.missing_predictions.detections) if 'missing_predictions' in sample and sample.missing_predictions else 0
                
                records.append({'Model': name, 'Predictions': existing, 'Status': 'existing'})
                records.append({'Model': name, 'Predictions': missing, 'Status': 'missing'})

        df = pd.DataFrame(records)

        # Plot setup
        plt.figure(figsize=(18, 8))
        sns.set(style="whitegrid")

        ax = sns.boxplot(x="Model", y="Predictions", hue="Status", data=df,
                        palette={"existing": "blue", "missing": "red"},
                        showfliers=False)

        plt.title("Boxplot of Existing and Missing Predictions per Sample", fontsize=14)
        plt.xlabel(r"\textbf{Model}")
        plt.ylabel(r"\textbf{Predictions per Sample}")
        plt.xticks(rotation=45)
        plt.legend(title=r"\textbf{legend}")

        plt.tight_layout()
        plt.savefig("plots/boxplot.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        
    def plot_roc_curves(self, li_models, li_names, title='', include_missing=False):
        from sklearn.metrics import roc_curve, auc

        plt.figure(figsize=(10, 6))

        li_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

        for ds, name, col in zip(li_models, li_names, li_colors):
            li = []
            y_true = []
            y_score = []
            print(name)

            for sample in ds:
                if 'detections' in sample and sample.detections:
                    for detection in sample.detections.detections:

                        
                        if 'eval' in detection and detection.eval:
                            li.append((detection.confidence, detection.eval))
                            if detection.eval == 'tp':
                                y_true.append(1)
                            elif detection.eval == 'fp':
                                y_true.append(0)
                        else:
                            print(detection)
                        y_score.append(detection.confidence)
                        # print(detection)
                        # print(f'conf = {detection.confidence}, eval = {detection.eval}')
            
                if include_missing and 'missing_predictions' in sample and sample['missing_predictions']:
                    for detection in sample.missing_predictions.detections:
                        y_true.append(0)
                        y_score.append(detection.confidence)

            print(f'length true = {len(y_true)}, len confidences = {len(y_score)}')
            # print(f"True positive = {total_tp}, False positive = {total_fp}, False negative = {total_fn}")
            # print(len(y_true), ', ', len(y_score))
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            # print(fpr)
            # Plot the ROC curve

            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
        plt.xlabel(r'\textbf{False Positive Rate}')
        plt.ylabel(r'\textbf{True Positive Rate}')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.savefig("plots/roc_"+ title +".pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def expected_calibration_error(self, confidences, accuracies, M=10):
        # uniform binning approach with M number of bins
        bin_boundaries = np.linspace(0, 1, M + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        # get max probability per sample i
        confidences = np.array(confidences)
    

        # get a boolean list of correct/false predictions
        accuracies = np.array(accuracies)

        ece = np.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # determine if sample is in bin m (between bin lower & upper)
            in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
            # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            prob_in_bin = in_bin.mean()

            if prob_in_bin.item() > 0:
                # get the accuracy of bin m: acc(Bm)
                accuracy_in_bin = accuracies[in_bin].mean()
                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = confidences[in_bin].mean()
                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
        return ece

    def plot_calibration_plot(self, li_models, li_names, title='', include_missing=False):
        
        from sklearn.calibration import calibration_curve

        li_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

        # Plot the calibration curve with error bars
        plt.figure(figsize=(10, 6))

        for ds, name, col in zip(li_models, li_names, li_colors):
            print(name)
            y_true = []
            confidence_scores = []
            for sample in ds:
                    if 'detections' in sample and sample.detections:
                        for detection in sample.detections.detections:

                            if 'eval' in detection and detection.eval:
                                if detection.eval == 'tp':
                                    y_true.append(1)
                                elif detection.eval == 'fp':
                                    y_true.append(0)
                                confidence_scores.append(detection.confidence)
                            else:
                                print(detection)

                    if include_missing and 'missing_predictions' in sample and sample.missing_predictions:
                        for detection in sample.missing_predictions.detections:
                            y_true.append(0)
                            confidence_scores.append(detection.confidence)
                        
            prob_true, prob_pred = calibration_curve(y_true, confidence_scores, n_bins=10)

            # Compute the number of samples in each bin (for error calculation)
            n_samples = [sum((np.array(confidence_scores) >= bin_left) & (np.array(confidence_scores) < bin_right)) 
                        for bin_left, bin_right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:])]

            # Calculate the error bars (binomial standard error)
            errors = [np.sqrt(p * (1 - p) / n) if n > 0 else 0 for p, n in zip(prob_true, n_samples)]

            prob_true_percent = np.array(prob_true) * 100
            errors_percent = np.array(errors) * 100
            prob_pred_percent = np.array(prob_pred) * 100

            ece = self.expected_calibration_error(confidence_scores, y_true)
            print(f'ECE for model "{name}": {ece[0]}')
            print(len(y_true)==len(confidence_scores))
                  
            
            plt.errorbar(prob_pred_percent, prob_true_percent, yerr=errors_percent, marker='o', linestyle='-', color=col, capsize=5, label = name)
            plt.plot(prob_pred_percent, prob_true_percent, marker='o', linestyle='-', color=col)

        plt.plot([0, 100], [0, 100], color='gray', linestyle='--', label = r"perfect calibration")  # Perfectly calibrated line
        plt.xlabel(r'\textbf{Mean confidence (\%)}')
        plt.ylabel(r'\textbf{Accuracy (\%)}')
        plt.title(title)
        plt.legend(loc='upper left')
        plt.savefig("plots/calibration_"+ title +".pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def get_sample_most_predictions(self, li_datasets, label):
        sample_with_most_predictions = None
        max_length = 0
        for sample1, sample2, sample3 in zip(*li_datasets):
            total_confidence = 0.0
            total_samples = []

            if label in sample1 and sample1[label]:
                total_samples.extend(sample1[label].detections)
            if label in sample2 and sample2[label]:
                total_samples.extend(sample2[label].detections)
            if label in sample3 and sample3[label]:
                total_samples.extend(sample3[label].detections)

            # Calculate total confidence score for all predictions of the sample
            for prediction in total_samples:
                total_confidence += prediction.confidence

            if len(total_samples) > max_length:
                max_length = len(total_samples)
                sample_with_most_predictions = sample1.id
                
        return sample_with_most_predictions
    
    def get_sample_with_extreme_confidence(self, li_datasets, label, highest = True):

        maxmin_average_confidence = 0.0 if highest else 1.0
        sample_best = None

        # Iterate over each sample in the dataset
        # for sample1, sample2, sample3 in zip(dataset_yolo_lvis, dataset_dino_lvis, dataset_gpt_lvis):
        for sample1, sample2, sample3 in zip(*li_datasets):
        # for sample1, sample2, sample3 in zip(dataset_yolo_japan, dataset_dino_japan, dataset_gpt_japan):
            total_confidence = 0.0
            total_samples = []

            if label in sample1 and sample1[label]:
                total_samples.extend(sample1[label].detections)
            if label in sample2 and sample2[label]:
                total_samples.extend(sample2[label].detections)
            if label in sample3 and sample3[label]:
                total_samples.extend(sample3[label].detections)

            # Calculate total confidence score for all predictions of the sample
            for prediction in total_samples:
                total_confidence += prediction.confidence

            # # Calculate average confidence score
            if len(total_samples) > 0:
                average_confidence = total_confidence / len(total_samples)
            else:
                average_confidence = 0 if highest else 1.0
                
            # Check if this sample has the highest average confidence score
            if ((average_confidence > maxmin_average_confidence and highest) or
                (average_confidence < maxmin_average_confidence) and not highest):
                maxmin_average_confidence = average_confidence
                sample_best = sample1.id

        return sample_best