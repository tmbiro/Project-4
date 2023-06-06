### Project 4 - Machine Learning: Ashley H, Fiona B, Katie B, Pratik S, Tifani B
# To Shroom or Not to Shroom?🍄 

> **One surprising development of the Covid-19 pandemic was the resurgence of interest in mushrooms among Americans.
As people looked for refuge from their homes in nature, many took on the practice of foraging mushrooms. This rise in interest has coincided with the use of Psilocybin mushrooms for medical and recreational use, in addition to discussions about how to combat climate change and cure world hunger.** 

## Contents
- [Purpose](#purpose)
    - [Is This Mushroom Edible or Poisonous?](#is-this-mushroom-edible-or-poisonous)
- [Datasource](#datasource)
- [Overview](#overview)
    - [Machine Learning Technologies](#machine-learning-technologies)
    - [Multi-Class Classification and Mushroom Species Identification](#multi-class-classification-and-mushroom-species-identification)
- [Description of Data Analysis](#description-of-data-analysis)
    - [Step 1: Data Cleaning and Preprocessing](#step-1-data-cleaning-and-preprocessing)
    - [Step 2](#step-2)
    - [Step 3](#step-3)
- [Results](#results)
    - [Findings](#findings)
    - [Analysis](#analysis)   
- [References](#references)

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/8d830f86-812a-4305-bc59-8a31a9ec96a8" />
</p>

## Purpose
If you were to walk around in your backyard right now, there is a good chance that you would find some sort of mushroom. Before handling these types of fungi, it is important to decipher whether or not they are poisonous. Misidentifying poisonous mushrooms can have severe consequences, as ingestion of toxic varieties can lead to illness, organ failure, or even death. We've decided to train a machine learning model that can accurately identify and differentiate toxic mushrooms from edible ones, ensuring the safety of individuals who forage or consume wild mushrooms.

### Is This Mushroom Edible or Poisonous?
This compelling question is one that our model can answer based on certain mushroom characteristics.

## Datasource
- [Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Secondary+Mushroom+Dataset)

## Overview

### Machine Learning Technologies
- Principal Component Analysis (PCA)
- Balancing Data if needed: Imbalanced-Learn w/ Imblearn in Python Random Over/Under Sampling.
- Binary Classification: edible or poisonous?
- Logistic Regression
- k-Nearest Neighbors
- Decision Trees
- Support Vector Machine
- TensorFlow

### Multi-Class Classification and Mushroom Species Identification
- k-Nearest Neighbors.
- Decision Trees.
- Random Forest.

## Description of Data Analysis
Data Analysis was done in a 3 step process:
#### - Step 1: Data Cleaning and Preprocessing
    - We began by importing our csv source to a jupyter notebook
    - Before modeling, we made sure to clean, normalize and standardize our data
        - When cleaning, we found dropping null values to be the most beneficial tactic since:
            - Whenever you have more % of null values better to drop the column or row
            - Whenever you have null values with outliers it's better to impute by median
            - Whenever you have null values with out outliers it's better to impute by mean
            - Whenever you have null values ok categorical values sometimes better to impute by mode.
            - On the occassional cases some times imputation of null values deals with using algorithmic imputation

#### - Step 2: Data Exploration and Visualization
    - In this step, we decided to create a barchart that displays how mushroom color influences whether the mushroom is poisonous or not.
    - We've also included a barchart that demonstrates how bruising and bleeding influences whether the mushroom is poisonous or not.
    - Additionally, our third barhcart checks how cap diameter influences whether the mushroom is poisonous or not.
    
#### - Step 3: Predictive Analyses

## Results

### Findings

### Analysis

## References
- [Mental Health and Substance Abuse](https://www.vox.com/2014/12/22/7424477/mushrooms-research)
- [Climate Change](https://www.nytimes.com/interactive/2022/07/27/climate/climate-change-fungi.html)
- [World Hunger](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10213758/) 
- [Legalization](https://www.vox.com/future-perfect/21509465/psychedelic-magic-mushrooms-psilocybin-medical-legalization-decriminalization-oregon-washington-dc) 
