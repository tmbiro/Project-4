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
        - [Deep Neural Net](#deep-neural-net) 
- [Conclusion and Next Steps](#conclusion-and-next-steps)
- [References](#references)

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/8d830f86-812a-4305-bc59-8a31a9ec96a8" />
</p>

## Purpose
If you were to walk around in your backyard right now, there is a good chance that you would find some sort of mushroom. Before handling these types of fungi, it is important to decipher whether or not they are poisonous. Misidentifying poisonous mushrooms can have severe consequences, as ingestion of toxic varieties can lead to illness, organ failure, or even death. We've decided to train a machine learning model that can accurately identify and differentiate toxic mushrooms from edible ones, ensuring the safety of individuals who forage or consume wild mushrooms.

#### Is This Mushroom Edible or Poisonous?
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

**Question 1:** How do mushroom cap characteristics influence mushroom toxicity?

> **Image 1a**
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/ed7d7671-4e54-4dc1-9ea3-49bab525a213" />
</p>

   - Results:** Buff mushrooms appear to be the least likely to poison you. You are almost guaranteed to be poisoned by a pink, green, orange, red, or yellow mushroom. Other colors appear to be a toss-up as to whether they are poisonous or not. Best to assume more mushrooms you find in the wild are poisonous.

> **Image 1b**
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/85eb14a8-42f4-4d64-9452-6c8b95efa0d9" />
</p>

   - **Results:** It appears that the bigger a mushroom cap is, the less likely it is to be poisonous.
  
**Question 2:** How do stem characteristics influence mushroom toxicity?

> **Image 2a** 
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/f5a54345-4058-464a-ae2b-d6cd5da6ae10" />
</p>

   - **Results:** Mushrooms with buff stems appear to almost always be edible. Mushrooms with white stems are also less likely to be poisonous. Stear clear of all other stem colors, however, as they are likely to be poisonous.

> **Image 2b**
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/0cae269a-de79-4573-910d-8c7f9e670236" />
</p>

   - **Results:** It appears that taller stemmed mushrooms are less likely to be poisonous.

> **Image 2c**
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/4517e7ad-6fc0-4325-aa9e-9032a0d45ba4" />
</p>

   - **Results:** Mushrooms with wider stems appear to be less likely to be poisonous.

**Question 3:** How do gill characteristics influence mushroom toxicity?

> **Image 3a**
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/998f8d15-52af-491b-afef-78dc356e57cb" />
</p>

   - **Results:** Mushrooms with white and buff gills appear to be less poisonous. Brown-, yellow-, red-, and pink-gilled mushrooms appear to be especially poisonous.

**Question 4:** How does bleeding and bruising influence mushroom toxicity?

> **Image 4a**
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/6cafbc03-b5a6-493a-ad6f-0e5c8a76e682" />
</p>

   - **Results:** It appears that you would have better luck not being poisoned by a mushroom that bleeds and/or bruises compared to one that doesn't, though you are likely to get poisoned regardless.

**Question 5:** How does habitat and season influence mushroom toxicity?

> **Image 5a**
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/18e956d4-9e24-437f-9bb9-331f18e62fac" />
</p>

   - **Results:** Urban mushrooms appear to be less poisonous, although their representation in the dataset is minimal. Leaf mushrooms also appear to be less poisonous. Mushrooms found in grasses or woods, however, appear more likely to be poisonous.

> **Image 5b**
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/534b0104-8376-4fe6-82c2-8cade7cc4e6d" />
</p>

   - **Result:** Most poisonous mushrooms can be found in the summer or autumn. You are less likely to find a poisonous mushroom in the Spring and Winter.

   ### Step 3: Predictive Analyses
    
   #### Data Transformation and Setup

**1.** First, we code the discrete variables to numerics   

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/eeca4337-e844-42aa-90bb-7b963a98ed2c" />
</p>


**2.** Then, we separate out our response variable (poisonous or not) from the predictor variables (everything else).

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/1dbaf2ed-3d71-47a9-a779-bfb214c88c66" />
</p>

**3.** If we don't apply scaling, the models will be more influenced by features such as cap shape and color and stem and gill color compared to features like bruises and whether the mushrooms has rings. This is because the former features have a greater impact due to their wider range of choices.

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/5c73e4fe-fa6e-4100-94f2-84aa0bfcf459" />
</p>

**4.** We can see from the correlation matrix that the features are mostly uncorrelated.

<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/0b61a794-3066-42b3-8e46-5d7fd91bbe47" />
</p>

   #### Principle Components Analysis (PCA)
   
   **1. Splitting the Data:** We start by splitting the dataset into training and testing sets using the train_test_split function. This allows us to have separate data to train our model and evaluate its performance.
   
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/e51900b7-b1cc-4a40-a71a-2dcf6955ce36" />
</p>

   **2. Standardizing the Data:** To ensure that all features are on the same scale, we create an instance of StandardScaler and fit it to the training data. Then, we use the fitted scaler to transform both the training and testing data. Standardization is important for many machine learning algorithms as it helps prevent certain features from dominating others due to differences in their scales.
   
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/87acc4ff-4b3d-4f3e-b6e3-6935f1369be2" />
</p>
   
   **3. Performing PCA:** We create a PCA object with n_components=2, indicating that we want to reduce the dimensionality of our data to two principal components. We then fit the PCA model to the scaled training data and transform both the training and testing data into the reduced feature space.

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/195b4804-c66f-4250-b3a9-b5ec00c5b82d" alt="alt text" width="whatever" height="whatever">

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/64f8c98b-0a63-4f5d-b371-02d51fe254a1" alt="alt text" width="whatever" height="whatever">

   **4. Visualizing the Transformed Data:** We plot the transformed data in a scatter plot, using the first principal component on the x-axis and the second principal component on the y-axis. The use of alpha=0.1 allows for better visualization when there are overlapping points, as it makes the points more transparent.

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/01206ea0-6710-440f-b76a-b09595a7be04" alt="alt text" width="whatever" height="whatever">


 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/942eea06-691b-4df3-bca2-a635013ee10f" alt="alt text" width="whatever" height="whatever">

   **5. Explained Variance Ratio:** We calculate and print the explained variance ratio for the first two principal components. The explained variance ratio tells us the proportion of the dataset's variance that is explained by each principal component. It provides insight into how much information is retained by each component.

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/941b61dc-3e04-466c-84d8-8761603c1922" alt="alt text" width="whatever" height="whatever">


 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/295cdfad-be12-44de-b7c5-674a0879baec" alt="alt text" width="whatever" height="whatever">
 
        - We can see that the first principle components analysis had a high accuracy at 87%; whereas, the second principle components analysis have relatively low accuracy at 8%.

   #### Random Forest 
   **1.** We checked for model accuracy.
   
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/4cb92918-771c-4d04-8b43-0d0307074ef4" />
</p>

   **2.** We calculated the confusion matrix.
 
 <p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/de737072-0b70-478a-8854-8fea29796241" />
</p>

   - True Positive (TP): 8446 True Negative (TN): 6760 False Positive (FP): 35 False Negative (FN): 27
        
   **3.** We visualized the features by importance.
   
 <p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/8fd308d7-eb50-44cf-a609-c10d7063c5ce" />
</p>
   
   #### Logistic Regression
   The goal of the logistic regression as a classification model was to see if it can predict the binary outcome: 
    
   - **Is this mushroom edible or poisonous?** 
   
   **1.** We fit a logistic regression model by using the training data (X_train and y_train).
   
 <p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/39e8ae64-44d0-4697-9310-c722ee15d6e1" />
</p>

   **2.** We printed the original accuracy score of the data and generated a confusion matrix for the model 
   
<p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/9370a720-81ec-4bd9-ba0a-3b3af88a84f2" />
</p>   

**Results** True Positive (TP): 6378 True Negative (TN): 2707 False Positive (FP): 4088 False Negative (FN): 2095
   
   **3.** We printed the classification report for the model
        
 <p align="center">
  <img src="https://github.com/Ahoust7/Project-4/assets/119274891/1e04dfcf-c2ab-4f3f-be8b-b04833de8697" />
</p>   

**Results:** Accuracy: 60% Class 0: Precision 56%, Recall 40%, F1-Score 47% Class 1: Precision 61%, Recall 75%, F1-Score 67% Weighted Average: Precision 59%, Recall 60%, F1-Score 58%

   **4.** We predicted a logistic regression model with over resampled training data

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/e151d3d8-bbc7-4c03-957d-768305b76e46" alt="alt text" width="whatever" height="whatever">

**Results:** True Positive (TP): 5011 True Negative (TN): 4038 False Positive (FP): 2757 False Negative (FN): 3462

   - Class 0 precision: 56%, recall: 40%, F1-score: 47% Class 1 precision: 61%, recall: 75%, F1-score: 67% Overall accuracy: 60% Weighted average precision: 59%, recall: 60%, F1-score: 58%

   **4.** We predicted a logistic regression model with under resampled training data

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/a6196e40-135a-4eb8-b8b9-f4d280fceda0" alt="alt text" width="whatever" height="whatever">

**Results:** True Positive (TP): 4952 True Negative (TN): 4078 False Positive (FP): 2717 False Negative (FN): 3521

   - Class 0 precision: 56%, recall: 40%, F1-score: 47% Class 1 precision: 61%, recall: 75%, F1-score: 67% Overall accuracy: 60% Weighted average precision: 59%, recall: 60%, F1-score: 58%


   **Findings**
   
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

   **1.** We defined features and set target vectors

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/80bdab34-cc45-4a6c-b41d-f1340125f151" alt="alt text" width="whatever" height="whatever">

   **2.**  We created a `StandardScaler` instance and scaled the training data

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/f5cbf964-5a5d-455d-91ed-d9d2b34ad1c8" alt="alt text" width="whatever" height="whatever">

   **3.** We calculated the confusion matrix and accuracy score

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/3279a509-d055-419e-9ac3-f52cf8459c92" alt="alt text" width="whatever" height="whatever">

   **Findings**
   
   - This produced an accuracy rating of 98% 
   - Model performed well which is likely because:
    - It is better able to handle that the mushroom features are not linearly related to whether the mushroom is edible or poisonous. 
    - As mentioned earlier we had outliers and missing data, and the decision tree model is not affected by this and is able to split features on the data accordingly. 
    - As seen in the image of the decision tree model the data is complex and deep.  It would be interesting to see the model applied to new data as more mushroom species are discovered. 

   #### Deep Neural Net
   
   **1.** Data Preparation: We split our preprocessed data into our features and target arrays & scaled the data

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/a4d688f0-dccf-4b75-b79e-d6a10a2da3d6" alt="alt text" width="whatever" height="whatever">

   **2.** Model Setup: We defined our model and deep neural net

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/471d14e3-db52-471b-a102-58b416cf8589" alt="alt text" width="whatever" height="whatever">

   **3.** We compiled and trained the model 

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/a84516fe-bc6a-4526-a04f-24ecfaa43eb9" alt="alt text" width="whatever" height="whatever">

   **4.** We evaluated the model using the test data

 <img src="https://github.com/Ahoust7/Project-4/assets/119274891/8687294a-ce27-4b03-8450-3cf8763960a0" alt="alt text" width="whatever" height="whatever">

**Explanation & Findings**

- To define the model in Step 2, we determined the number of input features we will use, how many hidden layers we were creating, and how many nodes each layer would use. For this model we chose 2 hidden layers with 100 and 75 nodes respectively, we also chose to use all the available features. The activation function for the two hidden layers was ReLU, which was chosen to explore non-linearity. For the output layer, we used sigmoid. 
- When compiling the model we chose the 'binary_crossentropy' function to calculate the loss because this is a binary classification model and this function computes the loss between true labels and predicted labels. Then when training the model, we found after a few attempts that 10 epochs gave use the highest accuracy with the lowest loss. 
- When evaluating the model in Step 4, it's evident that we've achieved 98.36% accuracy and 5.03% loss. 


## Conclusion and Next Steps

We were able to build several models that accurately predict the edibility or toxicity of mushrooms given the descriptive features in this dataset, however we believe that image identification of edible mushrooms would be significantly more helpful. So, our next step would be to use Machine Learning on mushroom photos to predict edibility. To this end we would explore both:

   **1.** A *Convolutional Neural Network Model* with Images of mushrooms to see if we can use images to identify if a mushroom is edible or poisonous.
    
   **2.** Using *Saliency Maps* to see if we can highlight which feature of the mushroom is relevant for classification of it being edible or poisonous.

Our group also discussed how changes in laws surrounding the cultivation and use of mushrooms containing psylocibin for therapeutic purposes will change the data landscape for mushrooms. Currently psylocibin is a controlled substance and data is not available or what is available is federally controlled. As the use of psylocibin mushrooms increases in western medicine and data is released. Adding the features of psilocybin mushrooms as a hallucinogenic classification category is a possibility.

## References
- [Mental Health and Substance Abuse](https://www.vox.com/2014/12/22/7424477/mushrooms-research)
- [Climate Change](https://www.nytimes.com/interactive/2022/07/27/climate/climate-change-fungi.html)
- [World Hunger](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10213758/) 
- [Legalization](https://www.vox.com/future-perfect/21509465/psychedelic-magic-mushrooms-psilocybin-medical-legalization-decriminalization-oregon-washington-dc)
- [Scientific Report Inspiration](https://www.nature.com/articles/s41598-021-87602-3)  
