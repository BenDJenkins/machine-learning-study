import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.metrics import r2_score


warnings.filterwarnings("ignore")

# Load Data
filepath = 'E:\GranuBeaker\savedata\gb_particledata'

# Load data
df = pd.read_feather(filepath, columns=None, use_threads=True)
df = df.dropna(axis=0)
headers = df.head()
print(headers)

# Extract feature and target arrays
X, y = df.drop(['no_particles', 'packing_fraction'], axis=1), df[['packing_fraction']]

# # Extract text features
# cats = X.select_dtypes(exclude=np.number).columns.tolist()

# # Convert to Pandas category
# for col in cats:
#    X[col] = X[col].astype('category')
   
# Split the data
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), random_state=1)

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

# Number of rounds
n = 100

model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=10, # Every ten rounds
   early_stopping_rounds=50
)

preds = model.predict(dtest_reg)

rmse = mean_squared_error(y_test, preds, squared=False)
r_squared = r2_score(y_test, preds)

print(f"RMSE of the base model: {rmse:.3f}")
print(f"R_squared of the base model: {r_squared}")

# Save model
file_name = "xgb_reg.pkl"
pickle.dump(model, open(file_name, "wb"))
