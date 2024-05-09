import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Loading the dataset
df = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df.head()

df.info()

df.describe()

df.columns

df.shape

# Checking for null values
df.isnull().sum()

"""From the above we can see that the data is clean from null values."""

df_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

df_categorical_columns

df_encoded = df.copy()

label_encoder = LabelEncoder()

for column in df_categorical_columns:
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

df_encoded

df_encoded.columns

sns.set(style="whitegrid")

df_encoded[df_encoded.columns].hist(bins=20, figsize=(30, 30))
plt.suptitle('Histograms of Features')
plt.show()

correlation_matrix = df_encoded.corr()

# Plot correlation heatmap using Seaborn
plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

X_ = df_encoded.drop(columns=['Attrition'])
Y_ = df_encoded['Attrition']

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_, Y_, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training set - Features:", X_train_.shape, " Target:", Y_train_.shape)
print("Testing set - Features:", X_test_.shape, " Target:", Y_test_.shape)

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=1000)

# Train the model on the training data
logistic_model.fit(X_train_, Y_train_)

# Predict the target variable for the test data
Y_pred_ = logistic_model.predict(X_test_)

# Evaluate the model
accuracy = accuracy_score(Y_test_, Y_pred_)
print("Accuracy:", accuracy)

# Get the classification report
print(classification_report(Y_test_, Y_pred_))



"""## Feature Engineering and Optimisation"""

df_new = df_encoded.drop(columns=['EmployeeCount','StandardHours', 'EmployeeNumber','Over18'])

correlation_matrix = df_new.corr()

# Plot correlation heatmap using Seaborn
plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

df_new['job_moninc'] = np.log(df_new['JobLevel'] * df_new['MonthlyIncome'])
df_new = df_new.drop(columns=['JobLevel', 'MonthlyIncome'])



df_new['salhike_perfrate'] = np.log(df['PercentSalaryHike']) / np.log(df['PerformanceRating'])
df_new = df_new.drop(columns=['PercentSalaryHike', 'PerformanceRating'])

df_new.columns



df_new["job_moninc_workyrs"] = np.log(df_new['job_moninc'] + df_new['TotalWorkingYears'])
df_new = df_new.drop(columns=['TotalWorkingYears','job_moninc'])

df_new["yrscomp_rol_man"] = df_new['YearsWithCurrManager'] * df_new['YearsInCurrentRole'] * df_new['YearsAtCompany']
df_new["yrscomp_rol_man"] = df_new["yrscomp_rol_man"]/np.mean(df_new["yrscomp_rol_man"])
df_new = df_new.drop(columns = ['YearsWithCurrManager', 'YearsInCurrentRole', 'YearsAtCompany'])

correlation_matrix = df_new.corr()

# Plot correlation heatmap using Seaborn
plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()



sns.set(style="whitegrid")

df_new[df_new.columns].hist(bins=20, figsize=(30, 30))
plt.suptitle('Histograms of the Optimised Features')
plt.show()



"""## Splitting the dataset to train and test"""

X = df_new.drop(columns=['Attrition'])
Y = df_new['Attrition']

X.head()



from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training set - Features:", X_train.shape, " Target:", Y_train.shape)
print("Testing set - Features:", X_test.shape, " Target:", Y_test.shape)



"""## Using Random Forest"""

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(n_estimators=100)

# Train the model on the training data
random_forest_model.fit(X_train, Y_train)

# Predict the target variable for the test data
Y_pred_rf = random_forest_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)
print("Accuracy for Random Forest:", accuracy_rf)

# Get the classification report
print(classification_report(Y_test, Y_pred_rf))



"""## Using Logistic Regression"""

from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(max_iter=1000)

# Train the model on the training data
logistic_model.fit(X_train_scaled, Y_train)

# Predict the target variable for the test data
Y_pred = logistic_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Get the classification report
print(classification_report(Y_test, Y_pred))

"""## Using Desicion Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state = 42)

# Train the classifier on the training data
clf.fit(X_train, Y_train)

# Make predictions on the testing data
Y_pred_clf = clf.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(Y_test, Y_pred_clf)
print("Accuracy:", accuracy)

"""## From the above we can see that the logistic regression produces more accuracy than the other models"""







