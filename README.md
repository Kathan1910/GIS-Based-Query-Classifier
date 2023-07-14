# GIS-based Query Classifier

This project aims to develop a machine learning model that classifies user queries as **attribute**, **spatial**, or a **combination** based on the provided input.

## Problem Statement

The goal of the query classification project is to build a Machine Learning model that classifies a query as attribute, spatial, or a combination based on user input.

## Query Definitions

- **Attribute Query:** An attribute query focuses on retrieving information based on the characteristics or attributes of a geographic feature. It involves searching for data related to specific attributes or properties of the features. 

  - Example: "Find all buildings with a floor area greater than 1000 square meters."

- **Spatial Query:** A spatial query involves analyzing geographic data based on their spatial relationships. It aims to retrieve data based on their proximity, containment, or intersection with other features in the geographic space.

  - Example: "Identify all parcels within a 1-mile radius of a given point."

- **Combination Query:** A combination query combines both attribute and spatial queries to retrieve data based on specific attribute conditions within a defined spatial area.

  - Example: "Find all houses with more than 3 bedrooms within a 5-kilometer radius of a park."

## Architecture of the Project

The project follows the following steps:

## 1. Data Cleaning

- **Rename Columns (Text, Target):** Renames the columns of the dataset to "Text" and "Target" for better clarity.
- **Label Encoding on Target Column:** Converts the target labels into numeric values for model training.
- **Check Missing Values:** Identifies and handles missing values in the dataset.
- **Check Duplicate Records:** Identifies and removes duplicate records from the dataset.

## 2. Exploratory Data Analysis (EDA)

Conducts exploratory data analysis to gain insights into the dataset and its characteristics.

## 3. Feature Extraction from Existing Features

Extracts relevant features from the existing dataset to improve the classification model's performance.

## 4. Data Preprocessing

Preprocesses the textual data to prepare it for model training:

- **Lower Case:** Converts all text to lower case for consistency.
- **Tokenization:** Splits the text into individual tokens.
- **Handle Special Characters:** Removes or handles special characters in the text.
- **Handle Stop Words & Punctuation:** Removes stop words and punctuation marks from the text.
- **Stemming:** Applies stemming to reduce words to their base form.

## 5. Text Vectorization

Converts the preprocessed text data into numerical vectors for model training:

- **Bag of Words:** Represents the text data as a matrix of word counts.
- **TF-IDF:** Represents the text data as a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) values.
- **Bag of N-Grams:** Represents the text data as a matrix of word n-gram counts.
- **Word2Vec Word Embeddings:** Represents the text data using Word2Vec word embeddings.

## 6. Model Training

Trains the following classification models on the preprocessed and vectorized data:

- **Logistic Regression**
- **Support Vector Classifier**
- **Multinomial Naive Bayes**
- **Decision Tree Classifier**
- **K-Nearest Neighbors Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **Bagging Classifier**
- **Extra Trees Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**

## 7. Ensemble Method

Utilizes ensemble methods to improve model performance:

- **Voting Classifier**
- **Stacking Classifier**

## Prediction with Web App

Predicts the query classifications using a web app created with Streamlit.


## Usage

1. Prepare your dataset in the desired format (e.g., CSV, Excel).
2. Execute the data cleaning steps mentioned above.
3. Perform exploratory data analysis using the provided Jupyter Notebook.
4. Extract relevant features from the dataset.
5. Preprocess the textual data as described above.
6. Apply text vectorization techniques to convert the text into numerical representations.
7. Train the classification models on the preprocessed and vectorized data.
8. Utilize ensemble methods to improve model performance.
9. Use the provided web app to make predictions based on user input.

For more detailed instructions and usage examples, please refer to the [Query Classifier PowerPoint Presentation](Query_Classifier.pptx)
