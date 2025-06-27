

# 🎯 Aviation Safety Predictive Analysis

---

## 📜 Table of Contents
- [📖 Project Overview](#-project-overview)
- [📊 Dataset Collection \& Cleaning](#-dataset-collection--cleaning)
- [✨Exploratory Data Analysis](#exploratory-data-analysis)
- [⚙️ Methodology \& Analysis Pipeline](#️-methodology--analysis-pipeline)
- [📈 Results](#-results)
- [🚀  Conclusion](#--conclusion)
- [🏗️ Future Work](#️-future-work)
- [📌 Repository Organization](#-repository-organization)

---

## 📖 Project Overview

This project analyzes aviation accident data from the National Transportation Safety Board (NTSB) to better understand the factors behind serious incidents. In an industry where safety is paramount, uncovering even subtle patterns can make a real difference.

Our work involves three key tasks:

1. **Injury Severity Prediction**: Classify injury proportions into three categories (minor/serious/fatal)
2. **Aircraft Damage Assessment**: Predict aircraft damage category (minor/substantial/destroyed)
3. **Accident Forecasting**: Forecast future accident counts using time-series methods

By combining these approaches, the project aims to highlight the conditions most associated with severe outcomes. The insights gained can support data-informed decisions by regulators, manufacturers, and airline operators to improve safety practices, save lives, and prevent aircraft damage.


## 📊 Dataset Collection & Cleaning


The project uses the official NTSB aviation accident database, which includes detailed information on each incident—such as aircraft type, flight phase, location, and weather conditions.

Key preprocessing steps:

-   Handled missing/inconsistent values.
    
-   Engineered new features (e.g., time since last inspection).
    
-   Split data into training and test sets to ensure robust model evaluation.
- **Add more**

---

## ✨Exploratory Data Analysis

**Add EDA plots**

**Aviation accidents heatmap:**

![](img/heatmap.png)

[Click to view interactive map](https://raw.githack.com/TheErdosInstitute-Summer2025-Project/aviation-project/main/img/heatmap.html)




![](img/interactive_heatmap.png)

[Click to view interactive map](https://raw.githack.com/TheErdosInstitute-Summer2025-Project/aviation-project/main/img/interactive_heatmap.html)

---

## ⚙️ Methodology & Analysis Pipeline

The project follows these key steps:

-   **Feature Selection:** A Random Forest model is used to rank feature importance and reduce dimensionality while preserving key predictors.
    
-   **Model Comparison:** Trained multiple models on the reduced set, including:
    -   Bagged Decision Trees
    -   Random Forest
    -   Extra Trees
    -   Histogram Gradient Boosting
    -   XGBoost
        
-   **Hyperparameter Tuning:** Each model is optimized via cross-validation. Final selection balances performance, efficiency, and interpretability.

- **Add more**

We built a time series forecasting pipeline using an LSTM model to predict monthly accident counts based on past data. The model captures temporal trends in the data and generates future estimates, with mean absolute error (MAE) used to evaluate prediction accuracy.

![](img/time_series_pred.png)

The MAE is 16.65, which is pretty low since the number of monthly accidents is usually above 100.

---



## 📈 Results


---

## 🚀  Conclusion

Taken together, our models all appear to indicate that the features included across the various NTSB datasets are not particularly predictive of the severity of an aircraft incident, nor the proportion of serious and fatal injuries in aircraft involved in accidents. This analysis illustrates the need for more thorough data collection and entry going forward if the NTSB data is to be used to predict aviation accident severity. 

---

## 🏗️ Future Work

While this project provides a strong foundation, there are several directions for future work:

* **Advanced Feature Engineering:** Explore dimensionality reduction techniques like PCA or t-SNE to capture complex relationships between features.

* **Deploy as an Interactive Tool:** Create a web-based dashboard that allows stakeholders to explore the data and test "what-if" scenarios using the final model.

* **Add More**

## 📌 Repository Organization

The main scripts are in the top level 
 - 01_ntsb_table_join.py: Combines all data into a single table, drops data outside the scope of our problem, and does train/validation/test split.
 - 02_ntsb_data_cleaning.py: Cleans training, validation, and test data.
 - 03a_modeling_damage.py: Models damage proportions with several ensemble models, tunes hyperparameters via cross-validation, saves best parameters and model performances on train and validation sets to file.
 - 04a_modeling_injury_proportion_trainval.py: Models injury proportions with several ensemble models, tunes hyperparameters via cross-validation, saves best parameters and model performances on train and validation sets to file.
 - 04b_modeling_injury_proportion_trainval.py: Runs the best injury proportion model on the test set.
 
'data' directory contents:
  - Subdirectories with raw NTSB data, cleaned NTSB data, t-100 data
  - It also has two files explaining the NTSB data: a codemap of all variables in the NTSB database
  - Detailed definitions of how injuries and aircraft damage are categorized.
 
'results' directory contents:
  - Parameters of best models obtained from grid search cross-validation
  - Plots of final model performance
  - Plots of feature importance in final model
 
'deliverables' directory contents:
  - Executive summary
  - Presentation slides
  - Previous project checkpoints.
 
'jupyter_notebooks' directory:
 - Notebooks for exploratory data analysis, model testing, and preliminary versions of code that went into the scripts.
