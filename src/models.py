import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

def train_rf(X_train, X_test, y_train, y_test, sport_name):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_res, y_res)
    pred = model.predict(X_test)
    recall = recall_score(y_test, pred)
    print(f"[{sport_name}] Random Forest Recall: {recall:.4f}")
    return recall, model

def train_xgb(X_train, X_test, y_train, y_test, sport_name):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1, use_label_encoder=False)
    model.fit(X_res, y_res)
    pred = model.predict(X_test)
    recall = recall_score(y_test, pred)
    print(f"[{sport_name}] XGBoost Recall: {recall:.4f}")
    return recall, model

def train_lgb(X_train, X_test, y_train, y_test, sport_name):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_res, y_res)
    pred = model.predict(X_test)
    recall = recall_score(y_test, pred)
    print(f"[{sport_name}] LightGBM Recall: {recall:.4f}")
    return recall, model

class InjuryMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, X_test, y_train, y_test, sport_name):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    X_train_t = torch.FloatTensor(X_res.values)
    X_test_t = torch.FloatTensor(X_test.values)
    y_train_t = torch.FloatTensor(y_res.values).reshape(-1,1)
    y_test_t = torch.FloatTensor(y_test.values).reshape(-1,1)
    
    model = InjuryMLP(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        pred = (model(X_test_t) > 0.5).float()
        recall = recall_score(y_test, pred.numpy())
    print(f"[{sport_name}] PyTorch MLP Recall: {recall:.4f}")
    return recall, model
