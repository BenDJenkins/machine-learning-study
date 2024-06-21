import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import optuna
import numpy as np
from sklearn.metrics import r2_score

# Define the MLP model for regression
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output one value for regression
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def objective(trial):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    filepath = 'E:\GranuBeaker\savedata\gb_particledata'
    df = pd.read_feather(filepath, columns=None, use_threads=True)
    df = df.dropna(axis=0)
    X = df.drop(['no_particles', 'packing_fraction'], axis=1).to_numpy()
    y = df['packing_fraction'].to_numpy()
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(DEVICE)  # Reshape y to be a column vector

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    
    # Create DataLoader for batching
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Define the model
    input_size = X.shape[1]
    model = MLP(input_size, hidden_size).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    
    total_loss = 0
    preds = []
    true_vals = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            preds.append(outputs.cpu().numpy())
            true_vals.append(labels.cpu().numpy())
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    preds = np.vstack(preds)
    true_vals = np.vstack(true_vals)
    r2 = r2_score(true_vals, preds)
    return r2

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", storage="sqlite:///db.sqlite3", study_name="mlp_study")
    study.optimize(objective, n_trials=500, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
