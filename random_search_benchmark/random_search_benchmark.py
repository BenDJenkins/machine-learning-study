import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import NodeConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import time

# Load your dataset
filepath = 'E:\GranuBeaker\savedata\gb_particledata'
df = pd.read_feather(filepath, columns=None, use_threads=True)
df = df.dropna(axis=0)
X_val = df.drop(['no_particles', 'packing_fraction'], axis=1)
Y_val = df['packing_fraction'].to_numpy()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_val, Y_val, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert X_train and X_test to DataFrame for pytorch_tabular
train_data = pd.DataFrame(X_train, columns=X_val.columns)
train_data['target'] = y_train
test_data = pd.DataFrame(X_test, columns=X_val.columns)
test_data['target'] = y_test

# Define models and their hyperparameter search spaces
models = {
    'Stochastic Gradient Descent': {
        'model': SGDRegressor(max_iter=1000, tol=1e-3),
        'params': {
            'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive', 'huber', 'squared_error'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
        }
    },
    'PLSRegression': {
        'model': PLSRegression(),
        'params': {
            'n_components': [1, 2, 3, 4, 5]
        }
    },
    'Gaussian Process': {
        'model': GaussianProcessRegressor(),
        'params': {
            'alpha': [1e-10, 1e-2, 1e-1, 1, 10],
            'normalize_y': [True, False]
        }
    },
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10, 50, 100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Support Vector Machine': {
        'model': SVR(),
        'params': {
            'C': np.logspace(0, 4, 8),
            'gamma': np.logspace(-4, 0, 8),
            'kernel': ['linear', 'rbf'],
        }
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(),
        'params': {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.5, 0.7, 1.0]
        }
    },
    'MLP': {
        'model': MLPRegressor(max_iter=500),
        'params': {
            'hidden_layer_sizes': [(100, 100), (200, 50), (200, 100), (50, 50, 50), (100, 50, 100), (200, 100, 100)],  # [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 50), (200, 100)]
            'activation': ['tanh', 'relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    },
}

# Define number of random search iterations
n_iterations = [1, 3, 5, 10, 15, 20, 50, 100]

results = []

# Perform random search for each model
for name, model_info in models.items():
    model = model_info['model']
    param_dist = model_info['params']
    for n_iter in n_iterations:
        if name == 'NODE':
            for model_config in param_dist['model_config']:
                trainer_config = param_dist['trainer_config']
                tabular_model = TabularModel(
                    data_config=DataConfig(
                        target=['target'],
                        continuous_cols=X_val.columns.tolist()
                    ),
                    model_config=model_config,
                    optimizer_config=OptimizerConfig(),
                    trainer_config=trainer_config
                )
                train_times = []
                for _ in range(100):
                    start_time = time.time()
                    tabular_model.fit(train=train_data, test=test_data)
                    train_time = time.time() - start_time
                    train_times.append(train_time)
                
                avg_train_time = np.mean(train_times)
                std_train_time = np.std(train_times)
                
                y_pred = tabular_model.predict(test_data).detach().cpu().numpy()
                score = r2_score(y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'Iterations': n_iter,
                    'Mean Accuracy': score,
                    'Std Accuracy': 0,  # Placeholder since we aren't using cross_val_score
                    'Avg Train Time': avg_train_time,
                    'Std Train Time': std_train_time
                })
        else:
            if param_dist:  # Check if there are any parameters to search
                random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=5, random_state=42, n_jobs=-1)
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)
            
            train_times = []
            for _ in range(100):
                start_time = time.time()
                best_model.fit(X_train, y_train)
                train_time = time.time() - start_time
                train_times.append(train_time)
            
            avg_train_time = np.mean(train_times)
            std_train_time = np.std(train_times)
            
            # Cross-validation to get accuracy and standard deviation
            scores = cross_val_score(best_model, X_test, y_test, cv=5, scoring='r2')
            mean_accuracy = np.mean(scores)
            std_accuracy = np.std(scores)
            
            results.append({
                'Model': name,
                'Iterations': n_iter,
                'Mean Accuracy': mean_accuracy,
                'Std Accuracy': std_accuracy,
                'Avg Train Time': avg_train_time,
                'Std Train Time': std_train_time
            })
        
        # print(f'Model: {name}, interations: {n_iter}, avg accuracy: {test_accuracy}')

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('model_benchmark_results_time.csv', index=False)
