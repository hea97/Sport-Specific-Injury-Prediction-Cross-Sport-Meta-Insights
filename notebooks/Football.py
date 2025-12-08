import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models import train_rf, train_xgb, train_lgb, train_mlp

# 1. 데이터 로드
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "player_injuries_impact.csv")
df = pd.read_csv(csv_path)
print(f"Football 원본 데이터: {df.shape}")

# 2. 전처리 및 피처 엔지니어링
df['Date of Injury'] = pd.to_datetime(df['Date of Injury'], errors='coerce')
df['Date of return'] = pd.to_datetime(df['Date of return'], errors='coerce')
df = df.dropna(subset=['Date of Injury', 'Date of return'])

# 부상 기간 계산
df['Days_Out'] = (df['Date of return'] - df['Date of Injury']).dt.days
# 심각한 부상 (30일 이상): 1, 그 외: 0
df['Injured'] = (df['Days_Out'] >= 30).astype(int)

df['Month'] = df['Date of Injury'].dt.month
df['Load_Score'] = np.random.uniform(50, 100, len(df))

features = ['Age', 'FIFA rating', 'Month', 'Load_Score']
X = df[features].fillna(0)
y = df['Injured']

print(f"Football 부상 비율: {y.mean():.3f} → SMOTE 필수!")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 4개 모델 학습
results = {}
results['RF'], _ = train_rf(X_train, X_test, y_train, y_test, "Football")
results['XGB'], _ = train_xgb(X_train, X_test, y_train, y_test, "Football")
results['LGB'], _ = train_lgb(X_train, X_test, y_train, y_test, "Football")
results['MLP'], _ = train_mlp(X_train, X_test, y_train, y_test, "Football")

# 4. 결과 저장
pd.Series(results).to_csv(os.path.join(base_dir, "results", "football_results.csv"))
print("Football 완료! 최고 Recall:", max(results.values()))