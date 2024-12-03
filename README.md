# Wine Quality Prediction with Random Forest in PySpark

## Overview

This project involves developing a machine learning pipeline using PySpark to predict wine quality based on physicochemical features. The model utilizes Random Forest classification with hyperparameter tuning to improve prediction accuracy.

## Table of Contents

- [Setup](#setup)
- [Data](#data)
- [Steps](#steps)
  - [AWS Setup](#aws-setup)
  - [Local Environment Setup](#local-environment-setup)
  - [Training the Model](#training-the-model)
  - [Validating the Model](#validating-the-model)
  - [Performance Metrics](#performance-metrics)
- [Directory Structure](#directory-structure)
- [Next Steps](#next-steps)


## Setup

### AWS Setup

1. Launched two EC2 instances (Master and Worker) with the following configuration:
   - Ubuntu 20.04
   - t2.medium instance type
2. Installed Spark and Java on both instances:
3. Configured Spark environment variables in `~/.bashrc`:
   - Reloaded `~/.bashrc` using `source ~/.bashrc`.

4. Started Spark services:

5. Verified the setup by accessing the Spark UI on port 8080.

## Data

Two datasets (`TrainingDataset.csv` and `ValidationDataset.csv`) were used. These files contain physicochemical properties of wine and their corresponding quality scores.

## Steps

### AWS Setup

Configured AWS EC2 instances for Spark cluster setup (detailed above).

### Local Environment Setup

1. Created a project directory and set up the following structure:
2. Installed PySpark and necessary libraries:
3. Moved the datasets into the `data/` directory.

### Training the Model

1. Wrote the `train_model.py` script to train a Random Forest model on the training dataset.
2. Performed hyperparameter tuning to optimize the model.
3. Saved the trained model in the `models/` directory:

### Validating the Model

1. Wrote the `predict.py` script to:
- Load the trained model.
- Make predictions on the validation dataset.
- Calculate performance metrics (accuracy, precision, recall, F1-score).

2. Saved the predictions in the `data/` directory.

### Performance Metrics

Achieved the following results on the validation dataset:
- Accuracy: 0.5125
- Precision: 0.4898
- Recall: 0.5125
- F1-Score: 0.4999

## Directory Structure

cs643-wine-quality/
├── data/
│   ├── TrainingDataset.csv
│   ├── ValidationDataset.csv
│   ├── predictions_rf.csv
├── models/
│   └── tuned_rf_model/
├── src/
│   ├── train_model.py
│   ├── train_rf_tuning.py
│   ├── predict.py
├── docker/
├── docs/
├── logs/
├── venv/
├── requirements.txt
└── README.md

## Next Steps

- Experiment with additional features or feature engineering to improve model performance.
- Deploy the model using AWS or a containerized solution.
- Integrate the pipeline into a web application for real-time predictions.

## Acknowledgments

This project is part of the CS643: Programming Assignment 2 coursework.