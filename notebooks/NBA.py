import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models import train_rf, train_xgb, train_lgb, train_mlp
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "injuries_2010-2020.csv")
df = pd.read_csv(csv_path)  # NBA 부상 데이터 사용
print(f"NBA 원본 데이터: {df.shape}")

# 2. 간단 전처리 (실제 컬럼명에 맞게 수정)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df['Injured'] = df['Notes'].str.contains('out|missed|injured', case=False, na=False).astype(int)

# 피처 엔지니어링 (예시)
df['Month'] = df['Date'].dt.month
df['Days_Missed'] = df['Relinquished'].notna().astype(int) * 5  # 임의 가정
df['Load_Score'] = np.random.uniform(50, 100, len(df))  # 실제론 GPS 등 사용

features = ['Month', 'Load_Score', 'Days_Missed']  # 실제 컬럼 있으면 추가
X = df[features].fillna(0)
y = df['Injured']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"부상 비율: {y.mean():.3f} → SMOTE 필수!")

# 3. 4개 모델 폭발 학습
results = {}
results['RF'], _ = train_rf(X_train, X_test, y_train, y_test, "NBA")
results['XGB'], _ = train_xgb(X_train, X_test, y_train, y_test, "NBA")
results['LGB'], _ = train_lgb(X_train, X_test, y_train, y_test, "NBA")
results['MLP'], _ = train_mlp(X_train, X_test, y_train, y_test, "NBA")

# 4. 결과 저장
results_path = os.path.join(base_dir, "results", "nba_results.csv")
pd.Series(results).to_csv(results_path)
print("NBA 완료! 최고 Recall:", max(results.values()))