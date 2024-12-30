# Machine Learning Based Resistor Classification

This repository includes machine learning-based algorithms to classify 4- and 5-Band resistors.   
The models within this repository are built with TensorFlow (2.16.1) and were trained on a small dataset: [Kaggle Resistor Dataset](https://www.kaggle.com/datasets/barrettotte/resistors). Due to the limited size of the dataset, transfer learning was applied.

## Preprocessing

In order for the models to predict a certain resistor, the images must be preprocessed and resized.  
The preprocessing relies on the following tools:

- [rembg](https://github.com/danielgatis/rembg): For removing image backgrounds.
- A custom-trained [YOLOv11n-Segmentation ](https://docs.ultralytics.com/models/yolo11/) Model: Used for segmentation tasks.
- [OpenCV](https://github.com/opencv/opencv-python): For additional image processing.

Below is a representation of the preprocessing pipeline:

![Preprocessing Pipeline](preprocessing.png)

## Models

This repository offers multiple models to choose from, with the best model achieving an accuracy of **86.61%**.  
However, there is still room for improvement, and the preprocessing steps might be highly specific to the dataset.

#### Example (1.5kΩ ± 5%)

<p align="center" width="100%">
    <img width="75%" src="4B-1K5-T5.jpg">
</p>

![4B-1K5-T5-Prediction](4B-1K5-T5_prediction.png)



