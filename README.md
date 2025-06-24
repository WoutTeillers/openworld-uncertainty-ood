# Bachelors-project

This project is designed to test open-world models on three datasets, evaluating both existing and missing objects in the images. The goal is to run experiments using three different models across the datasets and analyze the results.

## Models

The following models are tested in this project:

* **YOLO-World**
* **Grounding Dino**
* **GPT-4o**

## Datasets

The experiment uses the following datasets:

* **LVIS**
* **Open Images**
* **Japanese Uncertainty Scenes**

## Project Structure

### Main Function

The main function runs the experiment for all three models on all three datasets.

### `src/dataloader`

* **Purpose** : Loading of the datasets, generating a list of classes, and applying filtering logic on the list of classes.

### `src/dino`

* **Purpose** : Code for running  **Grounding Dino** .

### `src/yoloworld`

* **Purpose** : Code for running  **YOLO-World** .

### `src/gpt`

* **Purpose** : Code for interacting with the  **GPT API** .

### `src/result_shower`

* **Purpose** : Contains a class for loading and plotting the results.

### `get_results_final.ipynb`

* **Purpose** : A Jupyter notebook to view the results processed in `src/result_shower`.

## Installation

Make sure all necessary packages installed as definied in requirement.txt

Make sure to have a valid OPENAI API key in environment variables.
