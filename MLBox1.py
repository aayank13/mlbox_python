import pandas as pd 
import numpy as np 
import sys
from ydata_profiling import ProfileReport
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV                      
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer,LabelEncoder, OrdinalEncoder

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import seaborn as sns 
import matplotlib.pyplot as plt 

test_size = 0.2
missing_handler_num = 0
fillna_Random = -1

algo_num = 0
algorithm_handler = ["logistic_regression", "linear_regression", "random_forest", "knn"]
algorithm = algorithm_handler[algo_num]

scaler_num = 0
scaler_handler = ["minmax", "standard", "robust"]
scaler_method = scaler_handler[scaler_num]


pca_n = 2


# data = pd.read_csv('placement-dataset.csv')
# data = pd.rea     d_csv('gender_classification_v7.csv')
data = pd.read_csv('Breast_Cancer.csv')



# Data Preprocessing

#missing value imputation
missing_handler_num = 0
missing_handler = ["mean", "median", "forwardfill", "backwardfill", "deletion", "custom_value"]

if data.isnull().sum().any() > 0:
    if missing_handler_num == 0:
        data.fillna(data.mean(), inplace=True)
    elif missing_handler_num == 1:
        data.fillna(np.median(data), inplace=True)
    elif missing_handler_num == 2:
        data.ffill()
    elif missing_handler_num == 3:
        data.bfill()
    elif missing_handler_num == 4:
        data.dropna()
    else:
        data.fillna(fillna_Random)

#dublicate data removal 
data_total_duplicated= data.duplicated().sum()
data.drop_duplicates(inplace=True)

#Outliers Removal  
# Outlier detection type selection
outlierDetectionType = int(input("Outlier Detection Type: 1. IQR, 2. 98th Percentile: "))
numerical_Data=data.select_dtypes(include=['number'])

if outlierDetectionType == 1:
    # IQR method
    Q1 = numerical_Data.quantile(0.25)
    Q3 = numerical_Data.quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds for outliers using the IQR method
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (numerical_Data < lower_bound) | (numerical_Data > upper_bound)

elif outlierDetectionType == 2:
    # 98th percentile method
    threshold = numerical_Data.quantile(0.98)

    # Identify outliers
    outliers = numerical_Data > threshold

else:
    print("Invalid choice for outlier detection type.")


outlier_num = 0
outlier_handler = ["trimming_percentile", "capping", "change_to_mean","IQR_trim"]
outlier = outlier_handler[outlier_num]

if outlier == 'trimming':
    lower_percent = 3   ############# to make dynamic ###############
    upper_percent = 97  ############# to make dynamic ###############
    lower_bound = np.percentile(data, lower_percent, axis=0)
    upper_bound = np.percentile(data, upper_percent, axis=0)
    X_scaled = np.clip(data, lower_bound, upper_bound)

elif outlier == 'capping':
    lower_bound = data.mean() - 3 * data.std()
    upper_bound = data.mean() + 3 * data.std()
    X_scaled = np.clip(data, lower_bound, upper_bound)

elif outlier == 'change_to_mean':
    x_mean = np.mean(data, axis=0)
    X_scaled = np.where((data < x_mean - 3 * data.std()) | (data > x_mean + 3 * data.std()),
                        x_mean, data)
    
elif outlier=='IQR_trim':
    Q1 = numerical_Data.quantile(0.25)
    Q3 = numerical_Data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    X_scaled = np.clip(data, lower_bound, upper_bound)

data.loc[:, numerical_Data.columns] = data


################ Transformation ##########################
#StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
Normalization_tech=Normalization_tech = {
    0: StandardScaler(),
    1: MinMaxScaler(),
    2: MaxAbsScaler(),
    3: RobustScaler(),
    4: Normalizer(),
    5: QuantileTransformer(),
    6: PowerTransformer()
}
Normalization_tech_choice=int(input("choose the normalization {0:'StandardScaler()',1:'MinMaxScaler()',2:'MaxAbsScaler()',3:'RobustScaler()',4:'Normalizer()',5:'QuantileTransformer()',6:'PowerTransformer():  "))

    
scaler = Normalization_tech[Normalization_tech_choice]
data_scaled = scaler.fit_transform(data)

print(data_scaled)

# Data Splitting
y = data.iloc[:, -1]
X = data.iloc[:, :-1]

y_dtype = y.dtypes

if pd.api.types.is_categorical_dtype(y_dtype):
    modeltype = 0
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y = y_encoded
else:
    modeltype = 1

ordencod= OrdinalEncoder()
x_encord=ordencod.fit_transform(X)
X = x_encord

if X.shape[1] == 1:
    X = X.values.reshape(-1, 1)




# PCA
pca = PCA(n_components=pca_n)
x_pca = pca.fit_transform(X_scaled)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=test_size, random_state=42)

# Model Training with GridSearchCV
def train_model(x_train, y_train, algorithm):
    global model
    if algorithm == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'newton-cg', 'lbfgs', 'saga']
        }
    elif algorithm == 'linear_regression':
        model = LinearRegression()
        param_grid = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
    elif algorithm == 'random_forest':
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif algorithm == 'knn':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

best_model = train_model(x_train, y_train, algorithm)

# Model Evaluation
y_pred = best_model.predict(x_test)

if isinstance(best_model, (LogisticRegression, RandomForestClassifier, KNeighborsClassifier)):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
elif isinstance(best_model, LinearRegression):
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)



def save_model(best_model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)

# def load_model(filename):
#     with open(filename, 'rb') as file:
#         model = pickle.load(file)
#     return model

 


save_model(model, 'trained_model.pkl')


# # Later, for deployment
# loaded_model = load_model('trained_model.pkl')



# ############## Ydata Profiling #######################33

data_report=ProfileReport(data, title="Data Report")
data_report.to_html()
data_report.to_file("report.html")
          
                            