
# Bank Term Deposit Classification

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification algorithms to predict whether a bank customer will subscribe to a term deposit.

The project also provides an interactive Streamlit application for model evaluation and customer prediction.

## 2. Dataset Description

Dataset Name: Bank Marketing Dataset

Source: UCI Machine Learning Repository

Dataset Link:
https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

Dataset Details:
- Total Records: 45,211
- Original Input Features: 16
- Features Used: 15
- Target Variable: y
- Classification Type: Binary Classification
- Target Classes: yes / no

The duration feature was excluded because it is only known after the marketing call.

## 3. Machine Learning Models

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine (LinearSVC)

## 4. Data Preprocessing

- Numerical features were standardized using StandardScaler.
- Categorical features were encoded using OneHotEncoder.
- The dataset was divided into 80% training and 20% testing data.
- Stratified sampling was used to preserve the target class distribution.
- Preprocessing was included inside each model pipeline to prevent data leakage.

## 5. Model Evaluation

The following evaluation metrics were calculated:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Results

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8316 | 0.7188 | 0.3518 | 0.5217 | 0.4203 | 0.3347 |
| K-Nearest Neighbors | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Gaussian Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.7578 | 0.7718 | 0.2684 | 0.6200 | 0.3746 | 0.2867 |

## 6. Model Observations

### Logistic Regression
Achieved the highest AUC score of 0.7722 and the highest recall of 0.6238. It showed good ability to distinguish between customers who subscribe and those who do not.

### Decision Tree
Achieved the highest F1 score of 0.4203 and MCC of 0.3347. It provided the strongest balance according to these two metrics.

### K-Nearest Neighbors
Achieved the highest accuracy of 0.8874 and precision of 0.5529. However, its recall was only 0.1975, meaning it missed many actual subscribers.

### Gaussian Naive Bayes
Achieved an accuracy of 0.8452 and an F1 score of 0.4123. It demonstrated competitive performance compared with the other models.

### Support Vector Machine
Achieved an AUC score of 0.7718, close to Logistic Regression. Its recall was 0.6200.

## 7. Overall Model Comparison and Conclusion

The Decision Tree model achieved the highest F1 score and MCC among the five models. Therefore, it is the leading candidate when considering these two measures together.

Logistic Regression achieved the highest AUC, while KNN achieved the highest accuracy.

The results demonstrate that accuracy alone is not sufficient for evaluating an imbalanced classification problem. Multiple evaluation metrics provide a more complete understanding of model performance.

## 8. Application Features

The Streamlit application provides:

- Interactive analytics dashboard
- Comparison of five machine learning models
- Evaluation metrics visualization
- Confusion matrix
- Classification report
- Customer subscription prediction
- CSV test dataset upload and evaluation

Uploaded test data is used for evaluation. The application does not retrain the saved models.

## 9. GitHub Repository

https://github.com/TiwariAnshika20/Bank_Term_Deposit_Classification.git

## 10. Live Streamlit Application

https://banktermdepositclassification-ali9bjnrutaaogawr7wubt.streamlit.app/

## 11. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
