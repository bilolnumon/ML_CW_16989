Obesity Level Estimation Based on Eating Habits and Physical Condition
Project Overview


This project predicts obesity levels based on individuals’ eating habits and physical condition. 
It demonstrates the complete machine learning workflow: 
  data exploration, 
  preprocessing,
  model training, 
  evaluation, 
  deployment using a Streamlit web application.
  The dataset is sourced from UCI Machine Learning Repository: Estimation of Obesity Levels.


Table of Contents:
  
  Dataset
  Project Structure
  Requirements
  Setup Instructions
  Jupyter Notebook Walkthrough
  Streamlit Application
  Models
  Evaluation
  Deployment
  Reproducibility
  ataset


The dataset contains features related to:
Physical Condition: Height, Weight, BMI, Physical Activity Frequency (FAF), Time Using Electronic Devices (TUE), etc.
Eating Habits: Number of meals per day (NCP), Consumption of High Calorie Foods (CH2O), etc.
Target Variable: Obesity level (NObeyesdad) with classes such as:
  Insufficient_Weight, 
  Normal_Weight, 
  Overweight_Level_I, 
  Overweight_Level_II, 
  Obesity_Type_I, 
  Obesity_Type_II, 
  Obesity_Type_III.
The dataset contains both raw and synthetic data to improve model generalization.

Project Structure
ML_CW_16989/
│
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── notebooks/
│   └── obesity.ipynb           # Practical part: Data Load, EDA, Preparition, Model Training and Evolution
│
├── streamlit_app/
│   └── obesity_app.py          # Multi-page Streamlit app for exploration, preprocessing, inference, and evaluation
│
├── requirements.txt            # All dependencies
├── README.md
└── .gitignore


Requirements
  Python >= 3.8
  Pandas
  Numpy
  Scikit-learn
  Matplotlib
  Seaborn
  Streamlit
  Joblib


All required packages are listed in requirements.txt. Install them using:
pip install -r requirements.txt


Setup Instructions
If you do not have Python in your computer, it is required
Install every library inside requirements.txt
Install Jupyter and Python on VSCode
Start running codes one by one 


VSCode obesity.ipynb includes:
  Load Dataset
    Load CSV data into a Pandas DataFrame.
    Inspect the shape, columns, and summary statistics.
  Exploratory Data Analysis (EDA)
    Statistical summary for numeric and categorical features.
    Correlation matrix heatmap for numeric features.
    Boxplots, scatter plots, and histograms for feature visualization.
  Data Preparation
    Remove duplicates and handle missing values.
    Detect and remove outliers using IQR method.
    Ensure no negative values for numeric features.
    Feature engineering: BMI calculation.
    Dataset split into training, validation, and test sets (stratified).
  Model Training
    Train at least three models:
      Logistic Regression
      Random Forest
      K-Nearest Neighbors (KNN)
    Hyperparameter tuning using GridSearchCV.
  Model Evaluation
    Accuracy, classification report, and confusion matrices for each model.
    Comparison of models on the test set.
  
  
Streamlit Application
The Streamlit app (obesity_app.py) is multi-page:
  Data Page
    Preview original dataset and cleaned dataset.
    Remove duplicates and outliers.
  EDA Page
    Visualizations: statistical summary, histograms, scatter plot, and correlation matrix.
  Preprocessing Page
    Select target column, numeric/categorical columns.
    Choose imputation, encoding (OneHot or Label), and scaling options.
    Split data into train, validation, and test sets.
  Train & Tune Page
    Train models with optional hyperparameter tuning.
    Logistic Regression, Random Forest, and KNN supported.
  Inference Page
    Input single-sample values for prediction.
    Compute BMI and give obesity classification.
  Evaluation Page
    Compare models with accuracy, classification reports, and confusion matrices.


Models
  | Model               | Purpose                 | Notes                             |
  | ------------------- | ----------------------- | --------------------------------- |
  | Logistic Regression | Baseline classifier     | GridSearchCV tuning optional      |
  | Random Forest       | Ensemble method         | Handles nonlinear patterns        |
  | K-Nearest Neighbors | Instance-based learning | Sensitive to scaling and outliers |


Evaluation
  Metrics used:
    Accuracy
    Precision, Recall, F1-Score
    Confusion Matrix
    Models evaluated on test set to ensure unbiased comparison
  Deployment
    Fully interactive Streamlit app.
    Users can explore data, apply preprocessing, train models, perform inference, and evaluate models.
    Supports multi-page navigation for clear separation of workflow.
  Reproducibility
    All code is reproducible with requirements.txt.
    Dataset included in data/ folder.
    Random seeds used for train-test splits and model training.


Version Control
  Weekly commits maintained for all notebook and app development.
  Git repository contains history of preprocessing, model tuning, and app updates.
