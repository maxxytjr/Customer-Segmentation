# Customer-Segmentation
Customer Segmentation and Prediction for Arvato Financial Solutions

# Background
This project is modeled after a real-world business problem posed by Arvato Financial Solutions. The business owner is a mail-order company who wants to determine how they can acquire new customers more efficiently and effectively. 

This was achieved by analyzing the attributes and demographic information of existing customers and the rest of the population in Germany, begging the question: *what makes a customer or a non-customer?*.

The insights gained could be used to motivate marketing strategies targeted at specific clusters of the population, instead of reaching out to the whol population of Germany.

# Datasets

Two datasets were used for segmentation: one that describes attributes (age, income band, neighborhood characteristics) of the entire German population, and another dataset with the same attributes (features) that represents the customers.

For the supervised learning portion (prediction of potential customers), a dataset with similar features was used, albeit with a target variable to evaluate on.

Further details of the dataset cannot be shared due to license agreements.

# Solution Methodology

## Population Segmentation
  * Data cleaning by removing sparse columns, handling mixed datatypes and removing redundant features
  * Feature engineering by splitting existing features and creating new ones
  * One-hot encoding of categorical features
  * Data pre-processing to normalize values
  * Dimensionality reduction using *Principal Component Analysis*, generating a transformed dataset of principal components
  * k-means clustering of the transformed dataset to cluster data
  * Determination of customer clusters by comparing relative representations of each cluster in both datasets (general and customers)
  * Analysis of customer clusters to determine customer characteristics, by inspecting the top features and their weights in the principal components that have a large representation in the cluster


## Customer prediction
  * Data cleaning, normalization, and feature engineering with steps similar to the ones described above
  * Apply PCA and k-Means clustering to generate clusters of principal components
  * Slice dataset to contain only clusters that strongly represent customer and non-customers
  * Split data into training and test sets, specifying the `stratify` argument to ensure equal representation of the classes of the target variable in each set
  * Train `RandomForestClassifier`, `GradientBoostingClassifier` and `ElasticNet` classification algorithms from the *Scikit-Learn* libraries using **10-fold cross-validation**
  * Select the best model and its corresponding best hyperparameter combination based on its **cross-validation** score
  * Evaluate model on the test set, using the **Area under ROC Curve (AUROC) metric**
  
## Kaggle competition
  * Evaluate model on the dataset provided by Kaggle by predicting the probabilities of the target variable. The AUROC score was used as the scoring metric.


# Summary of Results (Supervised Learning)
  * Gradient Boosting Classifier determined to be the best model to use, based on its **cross-validation** score (0.98752). Random Forest Classifier came in very close (0.98749). However, Gradient Boosting Classifier took exceptionally long to train.
  * AUROC score of **0.722** on split test set
  * AUROC score of **0.79** on Kaggle test set
  
# Libraries Required
  * Python 3
  * Scikit-Learn
  * Pandas
  * Seaborn
  * MatPlotLib
  * Numpy
  * pickle
  
# Description of Files
  * README.md -- The file you are reading now
  * Arvato Project Workbook.ipynb -- Jupyter notebook containing the steps above from start to finish
  * final_model.pkl -- Model artifacts for the selected supervised learning model


# License and Acknowledgements
This project was completed as part of the Udacity Machine Learning Nanodegree. Arvato provided the data which cannot be publicly shared.


