# IBM-Employee-Attrition-Prediction
This folder contains code for predicting employee attrition using machine learning models. The dataset used for this prediction task is the IBM HR Analytics Employee Attrition Dataset, which is available on Kaggle that contains various features about employees and whether they have left the company (attrition) or not.

## Requirements
- Python 3
- Jupyter Notebook (optional)
- Required Python packages: numpy, pandas, matplotlib, seaborn, scikit-learn


## Usage
Run the Jupyter Notebook `predictiong_employee_attrition.ipynb`:
   ```
   jupyter notebook employee_attrition_prediction.ipynb
   ```
   Alternatively, you can run the Python script `predicting_employee_attrition.py` directly:
   ```
   python employee_attrition_prediction.py
   ```


## Additional Information
- The dataset used (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) contains various employee attributes such as age, job role, department, education, etc., along with the target variable `Attrition` indicating whether an employee has left the company or not.
- The code performs data exploration, preprocessing, feature engineering, and model training using machine learning algorithms such as Logistic Regression, Random Forest, and Decision Tree Classifier.
- Evaluation metrics such as accuracy, precision, recall, and F1-score are used to assess the performance of each model.
- The README file provides instructions for running the code and additional information about the project.

#### Note: Make sure all the required libraries or dependencies are satisified before executing the file.
