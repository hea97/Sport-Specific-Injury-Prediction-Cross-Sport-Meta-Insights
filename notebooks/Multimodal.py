import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models import train_rf, train_xgb, train_lgb, train_mlp

# 1. 데이터 로드
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "sports_multimodal_data.csv")
df = pd.read_csv(csv_path)
print(f"Multimodal 원본 데이터: {df.shape}")

# 2. 전처리 및 피처 엔지니어링
# 부상 위험도 타겟 변수 (injury_risk는 0/1 이진값)
df['Injury_Risk'] = df['injury_risk'].astype(int)

# 생리 신호 기반 피처 선택 (multimodal 데이터의 실제 컬럼 사용)
features = ['heart_rate', 'fatigue_index', 'workload_intensity', 'rest_period', 'training_duration']
X = df[features].fillna(df[features].median())
y = df['Injury_Risk']

print(f"Multimodal 부상 비율: {y.mean():.3f}")

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 4개 모델 학습
results = {}
results['RF'], _ = train_rf(X_train, X_test, y_train, y_test, "Multimodal")
results['XGB'], _ = train_xgb(X_train, X_test, y_train, y_test, "Multimodal")
results['LGB'], _ = train_lgb(X_train, X_test, y_train, y_test, "Multimodal")
results['MLP'], _ = train_mlp(X_train, X_test, y_train, y_test, "Multimodal")

# 5. 결과 저장
results_path = os.path.join(base_dir, "results", "multimodal_results.csv")
pd.Series(results).to_csv(results_path)
print("Multimodal 완료! 최고 Recall:", max(results.values()))