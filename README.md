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
    - [Step 2: Data Exploration and Visualization](#step-2-data-exploration-and-visualization)
    - [Step 3: Predictive Analyses](#step-3-predictive-analyses)
        - [Principle Components Analysis (PCA)](#principle-components-analysis-pca)
        - [Random Forest](#random-forest)
        - [Logistic Regression](#logistic-regression) 
        - [Decision Tree Model](#decision-tree-model)
- [Results](#results)
    - [Findings](#findings)
    - [Analysis](#analysis)   
- [Conclusion and Next Steps](#conclusion-and-next-steps)
- [References](#references)

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/8d830f86-812a-4305-bc59-8a31a9ec96a8" />
</p>

## Purpose
If you were to walk around in your backyard right now, there is a good chance that you would find some sort of mushroom. Before handling these types of fungi, it is important to decipher whether or not they are poisonous. Misidentifying poisonous mushrooms can have severe consequences, as ingestion of toxic varieties can lead to illness, organ failure, or even death. We've decided to train a machine learning model that can accurately identify and differentiate toxic mushrooms from edible ones, ensuring the safety of individuals who forage or consume wild mushrooms.

### Is This Mushroom Edible or Poisonous?
This compelling question is one that our model can answer based on certain mushroom characteristics.

## Datasource
- [Link to Dataset Files](https://mushroom.mathematik.uni-marburg.de/files/)
    - We are specifically using the `secondary_data_shuffled.csv` which can be found [here](https://mushroom.mathematik.uni-marburg.de/files/SecondaryData/secondary_data_shuffled.csv)

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
### Step 1: Data Cleaning and Preprocessing
   - We began by importing our csv source to a jupyter notebook
   - Before modeling, we made sure to clean, normalize and standardize our data
       - When cleaning, we found dropping null values to be the most beneficial tactic since:
        - Whenever you have more % of null values better to drop the column or row
        - Whenever you have null values with outliers it's better to impute by median
        - Whenever you have null values with out outliers it's better to impute by mean
        - Whenever you have null values ok categorical values sometimes better to impute by mode.
        - On the occassional cases some times imputation of null values deals with using algorithmic imputation

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/13a3653c-7c8d-4164-8c3e-bacee90e3399"> />
</p>

### Step 2: Data Exploration and Visualization
   - In this step, we decided to create a barchart that displays how mushroom cap and gill colors influences whether the mushroom is poisonous or not.

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/f2dc8073-1dfc-4b67-a355-e7fbd0736362" />
</p>

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/7288f3eb-c7b1-47f3-933f-35abe05a235f" />
</p>

   - We've also included a barchart that demonstrates how bruising and bleeding influences whether the mushroom is poisonous or not.

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/2a20c531-4f2e-4caa-839c-482ae74880d4" />
</p>

   - Our third barhcart checks how cap diameter influences whether the mushroom is poisonous or not.

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/4fb44a12-6d46-494a-89f7-81f50a216d17" />
</p>
    
   - Additionally, we visualized the effects of stem width, stem height, and stem color on whether a mushrooom is posionous or not. 

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/8425a734-bc29-474e-9aa6-b7cac5c5ea29" />
</p>

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/2190c771-8577-42a3-8f2e-31af588eb61f" />
</p>

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/1f9a855d-6272-4d28-ac40-0e0dcd764e87" />
</p>

### Step 3: Predictive Analyses
    
   #### Data Transformation and Setup
   
   #### Principle Components Analysis (PCA)
   
   **1. Splitting the Data:** We start by splitting the dataset into training and testing sets using the train_test_split function. This allows us to have separate data to train our model and evaluate its performance.
   
   **2. Standardizing the Data:** To ensure that all features are on the same scale, we create an instance of StandardScaler and fit it to the training data. Then, we use the fitted scaler to transform both the training and testing data. Standardization is important for many machine learning algorithms as it helps prevent certain features from dominating others due to differences in their scales.
   
   **3. Performing PCA:** We create a PCA object with n_components=2, indicating that we want to reduce the dimensionality of our data to two principal components. We then fit the PCA model to the scaled training data and transform both the training and testing data into the reduced feature space.
   
   **4. Visualizing the Transformed Data:** We plot the transformed data in a scatter plot, using the first principal component on the x-axis and the second principal component on the y-axis. The use of alpha=0.1 allows for better visualization when there are overlapping points, as it makes the points more transparent.
   
   **5. Explained Variance Ratio:** We calculate and print the explained variance ratio for the first two principal components. The explained variance ratio tells us the proportion of the dataset's variance that is explained by each principal component. It provides insight into how much information is retained by each component.
   
   #### Random Forest 
  - We checked for model accuracy.
   
 <p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/11b9b885-0abf-412b-9fb1-fdb1883548a8" />
</p>

  - We calculated the confusion matrix.
 
 <p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/8899e020-dd29-459b-8b53-7440d21b117b" />
</p>

  - We visualized the features by importance.
 
 <p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/f7e61b24-e97c-4e78-8c62-d10d76d6d28c" />
</p>
   
   #### Logistic Regression
   The goal of the logistic regression as a classification model was to see if it can predict the binary outcome: 
    - **Is this mushroom edible or poisonous?** 
   
   **Findings**
   - This produced an accuracy rating of 60% 
   - Over Sampled and Under Sampled LR Imbalanced Data Set 
    - Literature says it’s best to try both. 
    - **Over=** duplicates examples in the minority class (edible)
    - **Under=** merges examples in the majority class (poisonous) 60%
   - Model did not perform that well as can be seen by classification report which is most likely due to:
   - Data being too large.
   - There is no linear relationship b/w the outcome (edible and poisonous) and predicters (mushroom features)
   - 60% accuracy 

   #### Decision Tree Model
   The goal of the Decision Tree Model as a classification model was to see if it can predict the binary outcome:
    - **Is this mushroom edible or poisonous?** 

   **Findings**
   - This produced an accuracy rating of 98% 
   - Model performed well which is likely because:
    - It is better able to handle that the mushroom features are not linearly related to whether the mushroom is edible or poisonous. 
    - As mentioned earlier we had outliers and missing data, and the decision tree model is not affected by this and is able to split features on the data accordingly. 
    - As seen in the image of the decision tree model the data is complex and deep.  It would be interesting to see the model applied to new data as more mushroom species are discovered. 

## Results

### Findings
  - **Logistic Regression**
    - Works better when you can build a linear regression and we cannot do that with our data because there is so much, thus we can't get an accurate measure of regression.

### Analysis

## Conclusion and Next Steps

We were able to build several models that accurately predict the edibility or toxicity of mushrooms given the descriptive features in this dataset, however we believe that image identification of edible mushrooms would be significantly more helpful. So, our next step would be to use Machine Learning on mushroom photos to predict edibility. To this end we would explore both:

   **1.** A Convolutional Neural Network Model with Images of mushrooms to see if we can use images to identify if a mushroom is edible or poisonous.
    
   **2.** Using saliency maps to see if we can highlight which feature of the mushroom is relevant for classification of it being edible or poisonous.

Our group also discussed how changes in laws surrounding the cultivation and use of mushrooms containing psylocibin for therapeutic purposes will change the data landscape for mushrooms. Currently psylocibin is a controlled substance and data is not available or what is available is federally controlled. As the use of psylocibin mushrooms increases in western medicine and data is released. Adding the features of psilocybin mushrooms as a hallucinogenic classification category is a possibility.

## References
- [Mental Health and Substance Abuse](https://www.vox.com/2014/12/22/7424477/mushrooms-research)
- [Climate Change](https://www.nytimes.com/interactive/2022/07/27/climate/climate-change-fungi.html)
- [World Hunger](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10213758/) 
- [Legalization](https://www.vox.com/future-perfect/21509465/psychedelic-magic-mushrooms-psilocybin-medical-legalization-decriminalization-oregon-washington-dc)
- [Scientific Report Inspiration](https://www.nature.com/articles/s41598-021-87602-3)  
