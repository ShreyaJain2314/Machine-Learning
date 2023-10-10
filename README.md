# Machine-Learning

#ML-ALGOS

The code presents a data analysis and machine learning workflow using the Titanic dataset from Kaggle. It includes data preprocessing, visualization, and training and evaluation of machine learning models for predicting survival on the Titanic.

Here's a summary of what's happening in the code:

Data Loading and Initial Exploration: The Titanic dataset is loaded into a DataFrame named df using pd.read_csv. A heatmap is created to visualize the correlation between the features "Survived," "SibSp," "Parch," "Fare," and "Age."

Visualization: A bar plot is created to visualize the relationship between the number of siblings/spouses ("SibSp") and survival ("Survived").A FacetGrid is used to create histograms of the "Age" variable for both survivors and non-survivors.A bar plot is created to visualize the relationship between gender ("Sex") and survival ("Survived").A bar plot is created to visualize the relationship between passenger class ("Pclass"), gender ("Sex"), and survival ("Survived").Some exploratory data analysis (EDA) is performed on the "Embarked" variable.

Data Preprocessing: Missing values in the "Age" column are filled with random values generated within one standard deviation of the mean. Irrelevant columns ("PassengerId," "Name," "Cabin," and "Ticket") are dropped. Gender ("Sex") and port of embarkation ("Embarked") are mapped to numerical values. The features are separated into the variable x, and the target variable ("Survived") is separated into y.

Modeling and Evaluation: The dataset is split into training and testing sets using train_test_split. Standard scaling is applied to the feature data. Five different classifiers (Logistic Regression, Support Vector Classifier, Decision Tree Classifier, K-Nearest Neighbors Classifier, and Random Forest Classifier) are trained on the training data. Predictions are made using each model on the test data. Accuracy scores are calculated for each model using accuracy_score.

#SVM
The code appears is a data preprocessing and modeling workflow for a regression task using the "insurance.csv" dataset. Here's a summary of what's happening in the code:

Data Loading and Preprocessing: The dataset is loaded from a CSV file into a DataFrame called df. Two columns, "sex" and "smoker," with categorical values are one-hot encoded using pd.get_dummies to create binary (0 or 1) columns for each category. The "smoker" column is renamed to "smokers." Label encoding is applied to the "male" and "smokers" columns using LabelEncoder from scikit-learn. Data visualization is performed using Seaborn to explore relationships and distributions within the dataset. Various plots like countplots, boxplots, and scatterplots are used to visualize data.

Data Scaling: The feature data (xtrain and xtest) is standardized (scaled) using StandardScaler to have zero mean and unit variance.

Model Training: A Support Vector Regressor (SVR) from scikit-learn is initialized as regressor. The SVR is trained on the scaled training data (scaled_xtrain) and the corresponding target variable (ytrain) using regressor.fit.

Prediction: Predictions are made on the scaled training data (scaled_xtest) using the trained SVR model. The predictions are stored in the pred variable.

Data Exploration and Visualization: Throughout the code, various Seaborn visualization functions are used to explore the relationships between different columns in the dataset. These visualizations include countplots, boxplots, scatterplots, and a correlation heatmap.

Data Splitting: The dataset is split into training and testing sets using train_test_split from scikit-learn.


#Decision tree and Random Forest

The code is a  machine learning workflow for regression using the California housing dataset. Here's a breakdown of what's happening in the code:

Data Loading and Preprocessing: The California housing dataset is loaded using fetch_california_housing from scikit-learn. The data is stored in a DataFrame called data, and the target variable is named 'MedInc.' The target variable 'MedInc' is separated from the feature data.

Data Splitting and Scaling: The data is split into training and testing sets using train_test_split with a test size of 30% and a random seed of 42. Standardization (scaling) is applied to the feature data using StandardScaler to ensure that all features have zero mean and unit variance. The scaled training and testing data are stored in xtrain and xtest, respectively.

Decision Tree Regressor: A Decision Tree Regressor (DecisionTreeRegressor) is created as dtr. The Decision Tree Regressor is trained on the scaled training data (xtrain) and the corresponding target variable (y_train) using dtr.fit. Predictions are made on the scaled testing data (xtest) using dtr.predict. The mean squared error (MSE) between the predicted values (pred) and the true target values (y_test) is calculated using mean_squared_error.

Random Forest Regressor: A Random Forest Regressor (RandomForestRegressor) is created as rf_reg with 100 trees (estimators). The Random Forest Regressor is trained on the scaled training data (xtrain) and the corresponding target variable (y_train) using rf_reg.fit. Predictions are made on the scaled testing data (xtest) using rf_reg.predict.
The mean squared error (MSE) between the predicted values (y_pred) and the true target values (y_test) is calculated using mean_squared_error.

Model Evaluation: The code calculates the mean squared error (MSE) for both the Decision Tree Regressor and the Random Forest Regressor. MSE is a common metric for evaluating regression models, where lower values indicate better model performance. 

The primary goal of this code is to compare the performance of a Decision Tree Regressor and a Random Forest Regressor on the California housing dataset by evaluating their mean squared errors. The Random Forest Regressor is expected to perform better due to its ensemble nature, which combines multiple decision trees to improve predictive accuracy.
