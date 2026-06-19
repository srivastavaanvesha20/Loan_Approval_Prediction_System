# Loan Approval Prediction System

## Overview  
This project predicts whether a loan application is likely to be approved based on an applicant's financial background, credit history, and asset information. It follows a complete machine learning workflow—from cleaning and exploring data to training models and deploying the best one as a web app using Streamlit.

## Why This Project?  
Automating loan approval predictions helps lenders make faster, consistent, and data-driven decisions. It also gives applicants early feedback on their approval chances so they can improve their profile before applying.

## What's Included  

- **Data preprocessing**: Handling missing values, encoding categories, and cleaning inconsistencies  
- **Exploratory Data Analysis**(EDA): Visualizations and stats to uncover patterns  
- **Feature engineering**: Creating new features that boost model performance  
- **Model training & comparison**: Testing Logistic Regression, Random Forest, and XGBoost  
- **Model evaluation**: Measuring accuracy, ROC-AUC, and other key metrics  
- **Streamlit deployment**: A simple web app where users input details and get a prediction  

## Models Evaluated  

| Model               | Performance |
|---------------------|-------------|
| Logistic Regression | Good        |
| Random Forest       | Better      |
| **XGBoost**         | **Best** → Selected for deployment |

XGBoost delivered the strongest results and was chosen for the final app.

## Results (XGBoost)  

- **Accuracy**: 98.36%  
- **ROC-AUC**: 99.86%  


## Technologies Used  

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- XGBoost  
- Matplotlib  
- Streamlit  

## Project Structure  

```
Loan_Approval_Prediction/
├── Dataset/
├── Notebook/
├── Model/
├── Streamlit_App/
└── Documentation/
```

- **Dataset/**:         Raw and processed loan data  
- **Notebook/**:        EDA, feature engineering, and model training notebooks  
- **Model/**:           Saved XGBoost model file (`.pkl`)  
- **Streamlit_App/**:   Web app code (`streamlit_app.py`)  
- **Documentation/**:   Additional docs, reports, and usage guides  


