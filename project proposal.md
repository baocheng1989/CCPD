# CPS5990 Deep Learning Project - Chinese Car License Plate Recognition

## Team Member

|Student Name|Student ID|Email|
|------------|----------|-----|
|XinXin Qiu|1335785|<1335785@wku.edu.cn>|
|Cheng Bao|1335784|<baoc@kean.edu>|
|Bingyu Wang|1336571|<1336571@wku.edu.cn>|

## Problem Statement

As the non-human automation charging system is commonly used in public car parking lots nowadays, the recognition technique becomes crucial to determine whether the parking and charging of the car is done correctly and efficiently.

In this project, we want to implement the car plate recognition system using and modifying open-source codes as the baseline and 3rd dataset for training and validation. The core problem is image recognition with several special formats, in this case, the Chinese car license plate. As long as the format of the images, there are some things we should be concerned about, such as the angle or clarity of the car license plate image, as the car will enter the parking lot from various directions, and the core requirement for a car license recognition is to classify every vehicle correctly.

## Objective

Train a valid model to recognize the Chinese car license with many formats in picture, at least two formats (EV and non-EV), and two colors(blue for cars and yellow for trucks)

## Dataset Description

The training data source are from Kaggle or Github, sample are following, still searching:

- <https://www.kaggle.com/datasets/lyhue991/ccpd-sample>
- <https://github.com/detectRecog/CCPD?tab=readme-ov-file>
- <https://github.com/SunlifeV/CBLPRD-330k>

The dataset consists of Chinese license plates taken from cameras or CCTV, stored in JPEG format with different angles and resolutions.

## Methodology

Because the pictures in this dataset have different angles or resolutions, the dataset needs to be normalized and transformed first when it comes to the model training phase, maybe argument skill is used if it’s necessary.

In the training phase, we will use CNN models to achieve the goal, such as the YOLO v12 method. We will measure recognition accuracy in the validation phase to determine whether the model hit our goal.

## Tools & Libraries

The project is based on an open-source GitHub project; the link to the project is shown below. We will commonly use the deep learning techniques and tools such as PyTorch, torchvision and Jupyter Notebook for training the models, and Matplotlib or some Python libraries for validation and visualization.

## Expected Challenges

The challenge might be using unfamiliar CNN techniques, such as using the latest version of the YOLO method and correctly training the model. However, the dataset may need normalization or labeling, if required, which may cost more time than we expected. So we need some new knowledge preparation, paper reading and skill learning.

## Timeline

- Week 1: CCPD dataset searching and collection
- Week 2: Basement model finding
- Week 3: Necessary techniques preparation
- Week 4: Model training and validation
- Week 5: Project Presentation preparation

## Expected Outcome

We expect that in the end, we can train our own CCPD recognition model, and we can randomly take photos from cars outside, and it can give us the right answer.
