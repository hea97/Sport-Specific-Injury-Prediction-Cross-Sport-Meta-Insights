# 프로젝트 학습 로드맵 (학습 키워드 완전판)

이 문서는 Sport-Specific Injury Prediction 프로젝트를 완벽하게 이해하기 위해 학습해야 할 모든 주제와 키워드를 체계적으로 정리합니다.

---

## 1. 기초 프로그래밍 (선수지식)

### 1.1 Python 기초
- **변수와 자료형**: int, float, str, list, dict, tuple, set
- **제어문**: if-else, for, while, break, continue
- **함수**: def, return, *args, **kwargs, 람다 함수
- **예외 처리**: try-except, finally, raise
- **파일 입출력**: open(), read(), write(), with 문
- **객체지향 프로그래밍**: class, 상속, 다형성, 캡슐화

### 1.2 Python 고급 개념
- **리스트 컴프리헨션**: `[x*2 for x in range(10)]`
- **제너레이터**: yield, generator expression
- **데코레이터**: @property, @staticmethod, @classmethod
- **컨텍스트 매니저**: with 문의 원리
- **모듈과 패키지**: import, sys.path, __init__.py
- **가상환경**: venv, .venv 폴더의 역할

---

## 2. 데이터 처리 및 분석

### 2.1 Pandas (데이터 조작의 핵심)
- **DataFrame과 Series**: 2D/1D 데이터 구조
- **데이터 로드**: `pd.read_csv()`, `pd.read_excel()`
- **데이터 탐색**: `df.head()`, `df.info()`, `df.describe()`, `df.shape`
- **데이터 클린징**:
  - 결측치 처리: `df.isnull()`, `df.dropna()`, `df.fillna()`
  - 중복값 제거: `df.drop_duplicates()`
  - 데이터 타입 변환: `df.astype()`, `pd.to_datetime()`
- **데이터 필터링**: 불린 인덱싱, `df.loc[]`, `df.iloc[]`
- **데이터 정렬**: `df.sort_values()`, `df.sort_index()`
- **그룹화 집계**: `df.groupby()`, `df.agg()`, `df.apply()`
- **데이터 병합**: `pd.merge()`, `df.concat()`, `df.join()`
- **피벗과 메플트**: `df.pivot()`, `df.melt()`
- **시계열 처리**: `df.set_index()`, `df.resample()`

### 2.2 NumPy (수치 계산)
- **배열 생성**: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`
- **배열 인덱싱과 슬라이싱**: 다차원 배열 접근
- **벡터화 연산**: 반복문 대신 배열 연산
- **통계 함수**: `np.mean()`, `np.std()`, `np.min()`, `np.max()`, `np.median()`
- **선형대수**: `np.dot()`, `np.linalg.inv()`, `np.linalg.det()`
- **난수 생성**: `np.random.rand()`, `np.random.randn()`, `np.random.seed()`
- **배열 형태 변환**: `np.reshape()`, `np.flatten()`, `np.transpose()`

### 2.3 데이터 전처리 (프로젝트의 핵심)
- **결측치 전략**: 제거 vs 채우기 (평균, 중앙값, 앞/뒤 값)
- **아웃라이어 탐지**: IQR 방식, Z-score, 시각적 검사
- **정규화/표준화**:
  - Min-Max Scaling: (x - min) / (max - min) → [0, 1]
  - Z-score Normalization: (x - mean) / std → N(0, 1)
  - Robust Scaling: (x - median) / IQR
- **범주형 인코딩**:
  - One-Hot Encoding: 범주를 이진 열로 변환
  - Label Encoding: 범주를 숫자로 변환
  - Ordinal Encoding: 순서가 있는 범주 인코딩
- **텍스트 전처리**:
  - 소문자 변환: `.lower()`
  - 특수문자 제거: regex, `str.replace()`
  - 토큰화: 문자열을 단어로 분할
  - 불용어 제거: 의미 없는 단어(the, and, etc) 제거
- **특성 생성 (Feature Engineering)**:
  - 날짜에서 월, 요일, 분기 추출
  - 그룹 통계: 그룹별 평균, 합계
  - 다항 특성: $x^2$, $x \times y$
  - 비율과 차이: $x/y$, $x - y$
  - 시간차 특성: 이전 값과의 차이

---

## 3. 머신러닝 기초 개념

### 3.1 머신러닝 개요
- **지도학습 vs 비지도학습 vs 강화학습**
- **분류(Classification) vs 회귀(Regression) vs 군집(Clustering)**
- **부상 예측은 이진 분류 문제**:
  - 양성(Positive): 부상 발생 = 1
  - 음성(Negative): 부상 없음 = 0

### 3.2 모델 평가 지표 (성능 측정)
- **혼동행렬(Confusion Matrix)**:
  ```
  True Positive (TP): 올바르게 예측한 부상
  True Negative (TN): 올바르게 예측한 정상
  False Positive (FP): 잘못 예측한 부상 (거짓 경보)
  False Negative (FN): 놓친 부상 (가장 위험!)
  ```

- **정확도(Accuracy)**: (TP + TN) / 전체 → 전체 정답률
  - 문제: 부상이 6.5%일 때, 모두 정상 예측하면 93.5% accuracy!
  
- **정밀도(Precision)**: TP / (TP + FP) → 부상 예측 중 정답률
  - "부상 예측했을 때 얼마나 정확한가?"
  
- **재현율/민감도(Recall/Sensitivity)**: TP / (TP + FN) → 실제 부상 중 얼마나 잡는가?
  - **프로젝트 최우선 지표!** 부상을 놓치면 선수가 다침
  - Recall 90% = 100명 부상 중 90명 잡음, 10명 놓침
  
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - Precision과 Recall의 조화평균
  - 둘 다 중요할 때 사용
  
- **ROC 곡선과 AUC**: 임계값 변화에 따른 성능 곡선
  - AUC = 1: 완벽한 분류
  - AUC = 0.5: 무작위 분류
  
- **특이도(Specificity)**: TN / (TN + FP) → 정상을 정상이라 판정
  
- **클래스 불균형 처리**:
  - 현재 문제: 부상 6.5%, 정상 93.5% (14:1 비율)
  - 균형 잡히지 않은 데이터에서는 Accuracy 신뢰 불가능

### 3.3 데이터 분할 전략
- **Train/Test Split**: 80% 학습, 20% 평가
  - 순수한 평가 데이터로 최종 성능 측정
  
- **K-Fold Cross-Validation**: 5~10개 폴드로 나누어 검증
  - 모든 데이터가 검증 데이터가 될 수 있음
  - 더 안정적인 성능 추정
  
- **Stratified Split**: 클래스 비율 유지하며 분할
  - 부상 6.5% 비율을 Train/Test 모두에서 유지
  - 현재 프로젝트에서 사용: `stratify=y`

- **시계열 데이터 분할**: 시간 순서대로 분할
  - 미래 예측이 목표일 때 (과거 → 미래)
  - 현재는 사용 안 함 (스냅샷 데이터)

### 3.4 하이퍼파라미터 튜닝
- **Grid Search**: 모든 조합 시도 (느리지만 정확)
  ```python
  params = {'max_depth': [5, 10, 15], 'n_estimators': [100, 300, 500]}
  # 3 × 3 = 9개 조합 모두 테스트
  ```
  
- **Random Search**: 무작위 조합 시도 (빠르고 효과적)
  
- **Bayesian Optimization**: 확률 모델로 최적점 탐색 (가장 효율적)

### 3.5 과적합(Overfitting) vs 과소적합(Underfitting)
- **과적합**: 훈련 데이터에 너무 잘 맞춤 (테스트 성능 낮음)
  - 원인: 복잡한 모델, 너무 많은 특성, 적은 데이터
  - 해결: 정규화, 조기 종료, 데이터 증강
  
- **과소적합**: 훈련 데이터도 못 맞춤 (훈련/테스트 모두 낮음)
  - 원인: 너무 단순한 모델, 부족한 특성
  - 해결: 복잡한 모델, 더 많은 특성

---

## 4. 클래스 불균형 처리 (프로젝트의 핵심 기법)

### 4.1 문제 정의
- **현재 데이터**:
  - NBA: 부상 6.5%, 정상 93.5%
  - Football: 부상 45.1%, 정상 54.9%
  - Multimodal: 부상 5%, 정상 95%
  
- **문제점**: 모두 정상 예측하면 높은 정확도 → 무의미!

### 4.2 SMOTE (Synthetic Minority Over-sampling Technique)
- **원리**: 소수 클래스(부상) 샘플을 합성으로 생성
  
- **단계**:
  1. 부상 샘플 k개 최근접 이웃 찾기
  2. 무작위 이웃 선택
  3. 그 사이에 새로운 합성 샘플 생성
  4. 특성: 부상 = 100개 → 900개 (정상과 비율 맞춤)
  
- **장점**: 새로운 정보 추가, 정상 데이터 손실 없음
  
- **단점**: 합성 데이터 신뢰도, 오버피팅 위험
  
- **프로젝트에서**: `from imblearn.over_sampling import SMOTE`

### 4.3 대안 기법들
- **언더샘플링(Undersampling)**: 다수 클래스 줄이기
  - 장점: 빠른 학습
  - 단점: 정보 손실
  
- **가중치 조정(Class Weighting)**: 소수 클래스에 높은 가중치
  - `class_weight='balanced'` 파라미터 사용
  
- **임계값 조정(Threshold Tuning)**: 기본 0.5 대신 다른 값 사용
  - Recall 90% 달성 위해 임계값 낮춤

---

## 5. 트리 기반 머신러닝 모델

### 5.1 의사결정 트리 (Decision Tree)
- **개념**: 물음표 기반 분기 구조
  ```
  Age <= 30?
  ├─ Yes: Load > 70?
  │        ├─ Yes: 부상 위험 높음
  │        └─ No: 부상 위험 낮음
  └─ No: Position = Forward?
         ├─ Yes: 부상 위험 높음
         └─ No: 부상 위험 낮음
  ```

- **분할 기준**:
  - **Gini Index**: 노드의 불순도 측정
  - **엔트로피(Entropy)**: 정보 이론 기반 불순도
  
- **하이퍼파라미터**:
  - `max_depth`: 트리의 최대 깊이 (과적합 방지)
  - `min_samples_split`: 분할 최소 샘플 수
  - `min_samples_leaf`: 리프 노드 최소 샘플 수

- **장점**: 해석 가능, 빠른 학습
- **단점**: 과적합 경향, 데이터 작은 변화에 민감

### 5.2 Random Forest (랜덤 포레스트)
- **개념**: 여러 의사결정 트리의 앙상블
  
- **작동 원리**:
  1. Bootstrapping: 원본 데이터에서 복원추출로 부분집합 생성 (N개)
  2. 각 부분집합으로 독립적인 트리 학습
  3. 투표(분류) 또는 평균(회귀)으로 최종 예측
  
- **수학적 원리**:
  ```
  최종 예측 = (트리1 예측 + 트리2 예측 + ... + 트리N 예측) / N
  
  분류 (투표): 多數決
  Tree1 → 부상, Tree2 → 정상, Tree3 → 부상
  결과: 부상 (2:1 투표)
  ```
  
- **특성**:
  - 각 분할에서 무작위 피처 선택 (√p개 피처, p는 전체 피처 수)
  - 다양성 증가로 과적합 감소
  - 병렬화 가능 (n_jobs=-1) → 모든 트리 동시 학습
  - **Out-of-Bag (OOB) 오차**: Bootstrapping에서 사용 안 된 약 1/3 데이터로 검증
  
- **하이퍼파라미터**:
  - `n_estimators`: 트리 개수 (보통 100~500, 현재 300)
    - 많을수록 좋지만 수렴하는 지점 있음
  - `max_depth`: 각 트리의 최대 깊이 (무제한=None 보다 지정 권장)
    - 현재: 20 (부분적 과적합 허용)
  - `min_samples_split`: 노드 분할 최소 샘플 수
  - `min_samples_leaf`: 리프 노드 최소 샘플 수
  - `n_jobs`: -1 (모든 CPU 사용)
  
- **특징과 한계**:
  - **장점**:
    - 높은 성능 (베이스라인으로 우수)
    - 특성 중요도 계산 가능
    - 안정적 (극단값에 강건)
    - 병렬 처리로 빠른 학습
  - **단점**:
    - 깊은 비선형성 표현 어려움 (트리의 본질적 한계)
    - 센서 데이터의 상호작용 포착 불가능
    - 범주형 피처 많으면 효율성 저하

- **프로젝트 성능**:
  - NBA: 64.6% (기준선)
  - Football: 67.8% (작은 데이터에 강건)
  - Multimodal: 33.3% (센서 데이터에 부적합)

- **코드 예시**:
  ```python
  from sklearn.ensemble import RandomForestClassifier
  
  model = RandomForestClassifier(
      n_estimators=300,        # 300개 트리
      max_depth=20,            # 최대 깊이 20
      min_samples_split=2,     # 분할 최소 2개 샘플
      min_samples_leaf=1,      # 리프 최소 1개 샘플
      random_state=42,         # 재현성
      n_jobs=-1                # 모든 CPU 사용
  )
  
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  
  # 특성 중요도
  feature_importance = model.feature_importances_
  ```

### 5.3 XGBoost (Extreme Gradient Boosting)
- **개념**: 순차적 트리 부스팅으로 오류 개선
  
- **작동 원리**:
  1. 초기 예측 (무작위)
  2. 예측 오류 계산
  3. 이전 트리의 오류를 맞추는 새 트리 학습
  4. 가중 더하기로 다음 예측 개선
  5. 반복 (300번)
  
- **작동 원리 (상세)**:
  ```
  Step 0: 초기 예측 F₀(x) = log(p/(1-p)) (이진 분류에서 초기값)
  
  Step 1: 첫 번째 트리
    - 오류 = y - F₀(x) (실제값 - 예측값)
    - 이 오류를 맞추는 트리 h₁ 학습
    - 업데이트: F₁(x) = F₀(x) + η × h₁(x)  (η = learning_rate)
  
  Step 2: 두 번째 트리
    - 새로운 오류 = y - F₁(x)
    - 이를 맞추는 트리 h₂ 학습
    - 업데이트: F₂(x) = F₁(x) + η × h₂(x)
  
  ... (반복 300회)
  
  최종 예측: F₃₀₀(x) = F₀ + η×h₁ + η×h₂ + ... + η×h₃₀₀
  ```

- **핵심 개념 해설**:
  - **부스팅 (Boosting)**: 약한 학습기를 순차적으로 결합
    - RF는 병렬 (각 트리 독립)
    - XGB는 순차적 (이전 트리의 오류 고려)
  
  - **그래디언트 (Gradient)**: 손실 함수의 기울기
    - 오류를 미분으로 계산해 가장 가파른 방향으로 개선
  
  - **학습률 (Learning Rate)**: 각 단계에서 얼마나 이동할 것인가?
    - 낮은 값 (0.01): 천천히 학습, 안정적이지만 느림
    - 높은 값 (0.3): 빠르게 학습, 과적합 위험
    - 현재: 0.1 (균형점)
  
  - **정규화**: 복잡도 페널티로 과적합 방지
    - L1 페널티 (Lasso): |w| (가중치 크기)
    - L2 페널티 (Ridge): w² (가중치 제곱)
    - 트리 복잡도: 리프 노드 수, 깊이

- **손실 함수 (Loss Function)**:
  ```
  이진 분류: BCELoss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
  
  예시:
  실제: 부상 (y=1)
  예측: 0.8 (80% 부상)
  손실 = -[1 × log(0.8)] = 0.22 (작은 손실 = 좋은 예측)
  
  실제: 부상 (y=1)
  예측: 0.2 (20% 부상)
  손실 = -[1 × log(0.2)] = 1.61 (큰 손실 = 나쁜 예측)
  ```

- **하이퍼파라미터**:
  - `n_estimators`: 부스팅 반복 횟수 (현재: 300)
    - 많을수록 좋지만 수렴점 이후는 개선 없음
  - `max_depth`: 각 트리의 깊이 (현재: 8)
    - 깊을수록 복잡도 증가 → 과적합 위험
  - `learning_rate`: 학습률 (현재: 0.1)
    - 0.1은 중간 속도 (0.01~0.3 범위)
  - `subsample`: 각 반복에서 사용할 데이터 비율 (기본: 1.0)
  - `colsample_bytree`: 각 트리에서 사용할 피처 비율 (기본: 1.0)
  
- **특징과 한계**:
  - **장점**:
    - RF보다 우수한 성능 (부스팅 효과)
    - 정규화 내장 (과적합 방지)
    - 범주형 피처 자동 처리 가능
    - 누락된 값 처리 자동
  - **단점**:
    - 하이퍼파라미터에 민감 (조정 필수)
    - 순차적 학습으로 병렬화 어려움 (학습 느림)
    - Feature Importance 해석이 RF보다 어려움
    - 여전히 트리 기반의 비선형성 한계

- **프로젝트 성능**:
  - NBA: 66.0% (RF 64.6% + 1.4% 향상)
  - Football: 69.2% (RF 67.8% + 1.4% 향상)
  - Multimodal: 24.1% (RF 33.3% - 27.8% 악화!)
    - 왜 나빠짐? 센서 데이터의 비선형성을 트리가 포착 못함

- **코드 예시**:
  ```python
  from xgboost import XGBClassifier
  
  model = XGBClassifier(
      n_estimators=300,              # 300 부스팅 반복
      max_depth=8,                   # 깊이 8로 제한
      learning_rate=0.1,             # 학습률 10%
      subsample=1.0,                 # 전체 데이터 사용
      colsample_bytree=1.0,          # 전체 피처 사용
      random_state=42,
      n_jobs=-1,
      use_label_encoder=False,       # 최신 버전 호환성
      eval_metric='logloss'          # 평가 지표
  )
  
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  
  # 확률 예측 (부상 위험도)
  probabilities = model.predict_proba(X_test)[:, 1]
  ```

- **하이퍼파라미터 튜닝 예시**:
  ```python
  from sklearn.model_selection import GridSearchCV
  
  params = {
      'max_depth': [5, 8, 10, 12],
      'learning_rate': [0.01, 0.1, 0.3],
      'n_estimators': [100, 200, 300],
      'subsample': [0.7, 0.9, 1.0]
  }
  
  # 4 × 3 × 3 × 3 = 108개 조합 테스트
  gsearch = GridSearchCV(
      XGBClassifier(random_state=42),
      params,
      scoring='recall',  # Recall 최우선
      cv=5,              # 5-fold 교차 검증
      n_jobs=-1
  )
  gsearch.fit(X_train, y_train)
  best_model = gsearch.best_estimator_
  print(f"최적 Recall: {gsearch.best_score_:.4f}")
  ```

### 5.4 LightGBM (Light Gradient Boosting Machine) ⭐ 최우수
- **개념**: XGBoost의 가벼운 버전 (리프 기반 분할로 더 효율적)
  
- **XGBoost vs LightGBM 비교**:
  ```
  XGBoost (레벨 기반 / Level-wise):
  레벨 1:  [ ◆ ]
  레벨 2:  [ ◆ ◆ ◆ ◆ ]
  레벨 3:  [ ◆ ◆ ◆ ◆ ◆ ◆ ◆ ◆ ]
  → 모든 노드를 레벨별로 균형있게 성장
  → 메모리 더 사용, 깊은 트리에서 불필요한 분할
  
  LightGBM (리프 기반 / Leaf-wise):
         ◆
        / \
       ◆   ◆
      / \   \
     ◆   ◆  ◆
  → 손실 감소가 큰 리프부터 선택적으로 성장
  → 메모리 적게 사용, 같은 정확도를 빠르게 달성
  ```

- **분할 기준 수학**:
  ```
  각 노드에서 분할하는 기준:
  Gain = (손실값_왼쪽 + 손실값_오른쪽) - 손실값_원래
  
  XGBoost: 모든 가능한 분할점 시도 (비효율)
  LightGBM: Histogram 기반 분할 (효율적)
    - 연속 값을 그룹(bin)으로 나눔 (예: 10개 그룹)
    - 그룹 경계에서만 분할 탐색
    - 계산량 대폭 감소 (O(n) → O(b), b = bin 수)
  ```

- **LightGBM의 특성**:
  - **메모리 효율**: Histogram 기반 → 메모리 1/5~1/10 사용
  - **속도**: 병렬 처리 최적화 → XGBoost보다 10~20배 빠름
  - **정확도**: 리프 기반이 복잡한 상호작용 포착 효율적
  - **범주형 피처**: 자동 처리 (인코딩 불필요)
  - **GPU 지원**: nvidia-ml로 GPU 가속 가능
  
- **하이퍼파라미터**:
  - `n_estimators`: 부스팅 반복 (현재: 300)
  - `max_depth`: 트리 최대 깊이 (현재: 10)
  - `num_leaves`: 리프 노드 최대 수 (현재: 기본값 31)
    - LightGBM의 핵심 파라미터
    - num_leaves = 2^depth (균형 기준)
    - 현재 depth=10 → num_leaves 최대 1024 가능하지만 31로 제한
  - `learning_rate`: 학습률 (현재: 0.1)
  - `min_data_in_leaf`: 리프의 최소 샘플 수 (과적합 방지)
  - `verbose`: 로그 출력 (현재: -1로 억제)
  
- **특징과 한계**:
  - **장점**:
    - 전통 구조화 데이터에서 최고 성능
    - 매우 빠른 학습 속도
    - 메모리 효율적 (대규모 데이터 최적화)
    - 범주형 피처 자동 처리
    - 프로덕션 환경에서 검증됨 (Kaggle 우승 모델)
  - **단점**:
    - 소규모 데이터에서 과적합 경향
    - 깊은 트리로 인한 오버피팅 위험
    - 하이퍼파라미터 민감도 낮지만 튜닝 필수

- **프로젝트 성능** (최고! ⭐):
  - NBA: 68.6% (XGB 66.0% + 2.6% 향상) ⭐
  - Football: 71.1% (XGB 69.2% + 1.9% 향상) ⭐
  - Multimodal: 25.9% (센서 데이터에는 여전히 부적합)

- **코드 예시**:
  ```python
  from lightgbm import LGBMClassifier
  
  model = LGBMClassifier(
      n_estimators=300,           # 300 부스팅 반복
      max_depth=10,               # 최대 깊이 10
      num_leaves=31,              # 리프 노드 최대 31개
      learning_rate=0.1,          # 학습률 10%
      min_data_in_leaf=20,        # 리프의 최소 샘플 수
      random_state=42,
      verbose=-1,                 # 로그 억제
      n_jobs=-1,                  # 모든 CPU 사용
      objective='binary',         # 이진 분류
      metric='binary_logloss'      # 평가 지표
  )
  
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  
  # 확률 예측
  probabilities = model.predict_proba(X_test)[:, 1]
  ```

- **LightGBM의 장점 (실제 프로젝트)**:
  ```python
  # 1. 속도 비교
  import time
  
  start = time.time()
  lgb_model.fit(X_train, y_train)
  lgb_time = time.time() - start  # ~0.5초 (3개 데이터셋 병렬)
  
  start = time.time()
  xgb_model.fit(X_train, y_train)
  xgb_time = time.time() - start  # ~5초
  
  print(f"LGB가 {xgb_time/lgb_time:.1f}배 더 빠름")
  
  # 2. Feature Importance
  feature_importance = model.feature_importances_
  sorted_idx = np.argsort(feature_importance)[::-1]
  
  for idx in sorted_idx[:5]:
      print(f"{features[idx]}: {feature_importance[idx]:.4f}")
  ```

- **하이퍼파라미터 튜닝 (Football 같은 소규모 데이터)**:
  ```python
  params = {
      'max_depth': [5, 7, 10],
      'num_leaves': [20, 31, 50],
      'learning_rate': [0.05, 0.1, 0.2],
      'min_data_in_leaf': [10, 20, 30]
  }
  
  from sklearn.model_selection import GridSearchCV
  gsearch = GridSearchCV(
      LGBMClassifier(n_estimators=300, random_state=42),
      params,
      scoring='recall',
      cv=5,
      n_jobs=-1
  )
  gsearch.fit(X_train, y_train)
  print(f"최적 Recall: {gsearch.best_score_:.4f}")
  print(f"최적 파라미터: {gsearch.best_params_}")
  ```

### 5.5 피처 중요도 (Feature Importance)
- **개념**: 각 특성이 모델 예측에 얼마나 기여하는가?
  
- **계산 방식**:
  - **Gain**: 각 분할이 손실함수를 얼마나 감소시켰는가
  - **Split**: 각 특성이 몇 번 사용되었는가
  - **Cover**: 각 특성이 영향을 미친 샘플 수
  
- **사용 예**:
  ```
  LightGBM Feature Importance:
  Load_Score: 0.32 (가장 중요)
  Month: 0.28
  Position: 0.22
  Age: 0.18
  ```
  - Load_Score가 부상 예측에 가장 결정적

---

## 6. 신경망과 딥러닝

### 6.1 인공신경망 기초
- **뉴런(Neuron)**: 뇌의 신경세포 모방
  ```
  입력(x1, x2, ...) 
    → 가중치 곱하기 (w1, w2, ...)
    → 합산
    → 활성화 함수
    → 출력
  ```

- **가중치(Weight)**: 입력의 중요도
  - 학습 중에 자동 조정
  
- **편향(Bias)**: 상수항
  - 활성화 함수를 옮기는 역할

- **활성화 함수(Activation Function)**:
  - **Sigmoid**: 0~1 범위 (이진 분류)
  - **ReLU**: max(0, x) (은닉층 표준)
  - **Tanh**: -1~1 범위
  - **Softmax**: 다중 분류 확률 분포

### 6.2 신경망 아키텍처 (MLP)
- **구조**:
  ```
  입력층 (Input Layer): 5개 노드
    ↓
  은닉층 (Hidden Layer 1): 128개 노드 + ReLU
    ↓
  드롭아웃: 30% 제거 (과적합 방지)
    ↓
  은닉층 (Hidden Layer 2): 64개 노드 + ReLU
    ↓
  드롭아웃: 30% 제거
    ↓
  은닉층 (Hidden Layer 3): 32개 노드 + ReLU
    ↓
  출력층 (Output Layer): 1개 노드 + Sigmoid
    → 확률값 (0~1)
  ```

- **깊이 vs 폭**:
  - 깊이: 은닉층 개수 (비선형성 증가)
  - 폭: 각 층의 노드 수 (표현력 증가)
  - 현재: 깊이 3, 폭 128→64→32 (점진적 축소)

### 6.3 학습 과정 (상세)
- **순전파(Forward Pass)**:
  ```
  입력: x = [heart_rate, fatigue_index, workload, rest_period, training_duration]
  
  은닉층 1:
  z₁ = W₁ × x + b₁  (선형 변환)
  a₁ = ReLU(z₁)     (비선형 활성화)
  
  은닉층 2:
  z₂ = W₂ × a₁ + b₂
  a₂ = ReLU(z₂)
  
  은닉층 3:
  z₃ = W₃ × a₂ + b₃
  a₃ = ReLU(z₃)
  
  출력층:
  z₄ = W₄ × a₃ + b₄
  ŷ = Sigmoid(z₄)   (확률값 0~1)
  ```
  
- **손실 함수(Loss Function) - BCELoss**:
  ```
  L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
  
  예시 1: 실제 부상(y=1), 예측 0.8(80% 부상)
    L = -[1 × log(0.8) + 0 × log(0.2)]
    L = -(-0.223) = 0.223 (작은 손실 ✓)
  
  예시 2: 실제 부상(y=1), 예측 0.2(20% 부상)
    L = -[1 × log(0.2) + 0 × log(0.8)]
    L = -(-1.609) = 1.609 (큰 손실 ✗)
  ```
  - 손실 함수의 역할: 모델이 얼마나 틀렸는지 수량화

- **역전파(Backpropagation) - 가중치 업데이트**:
  ```
  Step 1: 손실 계산
    L_total = sum(L_i) / 배치크기
  
  Step 2: 각 가중치의 기울기 계산 (미분)
    ∂L/∂W₄ = 손실 관련 W₄의 영향도
    ∂L/∂W₃ = 손실 관련 W₃의 영향도
    ...
    ∂L/∂W₁ = 손실 관련 W₁의 영향도
  
  Step 3: 경사 하강법으로 가중치 업데이트
    W₄_new = W₄_old - learning_rate × ∂L/∂W₄
    W₃_new = W₃_old - learning_rate × ∂L/∂W₃
    ...
    W₁_new = W₁_old - learning_rate × ∂L/∂W₁
  
  핵심: 기울기가 음수면 W 증가, 양수면 W 감소 → 손실 방향 반대로 이동
  ```

- **최적화기(Optimizer) - Adam의 작동**:
  ```
  기본 SGD (문제점):
    W_new = W - lr × ∂L/∂W  (동일한 학습률)
    → 가파른 방향: 진동
    → 완만한 방향: 느린 수렴
  
  Adam (Adaptive Moment Estimation):
    m = 0.9 × m_old + 0.1 × ∂L/∂W  (모멘텀)
    v = 0.999 × v_old + 0.001 × (∂L/∂W)²  (제곱값)
    
    적응형 학습률: lr_adaptive = lr / (√v + ε)
    가중치 업데이트: W_new = W - lr_adaptive × m
  
  효과:
    - 모멘텀: 이전 기울기 고려 → 진동 감소
    - 제곱값 기반: 가파른 방향은 학습률 감소, 완만한 방향은 증가
    → 각 가중치마다 적응형 학습률 자동 조정 ✓
  ```

- **에포크(Epoch)와 배치(Batch)**:
  ```
  현재 설정: 50 에포크
  
  에포크 1:
    전체 데이터 4,344개를 배치로 나누어 학습
    - 배치 1: 샘플 1~32 학습 → W 업데이트
    - 배치 2: 샘플 33~64 학습 → W 업데이트
    - ...
    - 배치 136: 샘플 4,321~4,344 학습 → W 업데이트
  
  에포크 2~50: 반복
  
  배치 효과:
    - 작은 배치 (16): 자주 업데이트, 노이지지만 빠른 수렴
    - 큰 배치 (256): 안정적, 더 정확한 기울기, 느린 수렴
    - 현재: 기본값 (32)
  ```

- **손실 함수의 기울기 계산 (체인 룰)**:
  ```
  컴퓨터 자동으로 계산 (PyTorch의 autograd):
  
  손실 → 출력층 활성화 → W₄ → W₃ → W₂ → W₁
  
  역방향으로 미분 (체인 룰):
  ∂L/∂W₁ = ∂L/∂ŷ × ∂ŷ/∂z₄ × ∂z₄/∂a₃ × ... × ∂z₁/∂W₁
  
  복잡도: O(깊이) (깊을수록 계산량 증가, 깊이 소실/폭발 위험)
  ```

### 6.4 PyTorch 구현 (프로젝트 코드)
- **신경망 클래스 정의**:
  ```python
  import torch
  import torch.nn as nn
  
  class InjuryMLP(nn.Module):
      def __init__(self, input_size):
          super().__init__()
          self.net = nn.Sequential(
              # 입력층 → 은닉층 1 (5 → 128)
              nn.Linear(input_size, 128),
              nn.ReLU(),
              nn.Dropout(0.3),           # 30% 뉴런 비활성화
              
              # 은닉층 1 → 은닉층 2 (128 → 64)
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Dropout(0.3),
              
              # 은닉층 2 → 은닉층 3 (64 → 32)
              nn.Linear(64, 32),
              nn.ReLU(),
              
              # 은닉층 3 → 출력층 (32 → 1)
              nn.Linear(32, 1),
              nn.Sigmoid()               # 확률값 (0~1)
          )
      
      def forward(self, x):
          return self.net(x)
  ```

- **학습 루프 (간소화)**:
  ```python
  model = InjuryMLP(input_size=5)
  criterion = nn.BCELoss()                    # 손실 함수
  optimizer = optim.Adam(model.parameters(), lr=0.001)  # 최적화기
  
  for epoch in range(50):                     # 50 에포크
      optimizer.zero_grad()                   # 기울기 초기화
      
      # 순전파
      outputs = model(X_train_tensor)
      loss = criterion(outputs, y_train_tensor)
      
      # 역전파
      loss.backward()                         # 기울기 계산
      
      # 가중치 업데이트
      optimizer.step()                        # W_new = W - lr × ∂L/∂W
      
      if (epoch + 1) % 10 == 0:
          print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
  ```

- **PyTorch의 주요 기능**:
  ```python
  # 1. 자동 미분 (Autograd)
  x = torch.tensor([2.0, 3.0], requires_grad=True)
  y = x ** 2 + 3 * x
  y.sum().backward()  # ∂y/∂x 자동 계산
  print(x.grad)  # [7.0, 9.0] (2*2+3, 2*3+3)
  
  # 2. 데이터 로더 (배치 처리)
  from torch.utils.data import DataLoader, TensorDataset
  dataset = TensorDataset(X_train, y_train)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  for batch_x, batch_y in loader:
      # 배치 크기 32로 처리
      pass
  
  # 3. GPU 전송 (NVIDIA GPU 사용 시)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  X_train = X_train.to(device)
  ```

### 6.5 드롭아웃 (Dropout) - 정규화 기법
- **개념**: 학습 중 일부 뉴런을 무작위로 비활성화
  ```
  드롭아웃 = 30%
  
  훈련 중:
  입력 노드: [128개] → 30% 비활성화 → [약 90개 활성화]
  
  테스트 중:
  입력 노드: [128개] → 100% 활성화 (드롭아웃 미적용)
  가중치 × 0.7 (훈련 때의 비율 보상)
  ```

- **작동 원리**:
  ```
  마스크 벡터 m을 확률 p로 생성:
  m = [1, 0, 1, 1, 0, ...] (1=활성화, 0=비활성화)
  
  활성화: a'(t) = a(t) * m / (1-p)
  
  30% 드롭아웃의 의미:
    - 매 배치마다 다른 부분집합 네트워크 훈련
    - 네트워크 A, 네트워크 B, ... 등 수천 개 부분 네트워크의 조합
    - 앙상블 효과 (여러 모델의 평균)
  ```

- **효과**:
  - **과적합 방지**: 특정 뉴런에 의존하지 않도록 강제
  - **앙상블 효과**: 부분 네트워크들이 다양한 관점 학습
  - **공적응(Co-adaptation) 방지**: 뉴런들이 상호의존하지 않도록
  
- **프로젝트의 드롭아웃**:
  ```
  은닉층 1 (128) → 드롭아웃 30% → 약 90개
  은닉층 2 (64) → 드롭아웃 30% → 약 45개
  은닉층 3: 드롭아웃 없음 (출력 직전)
  ```

### 6.6 신경망 vs 트리 모델 비교
- **신경망의 비선형성**:
  ```
  트리 모델 (선형/다항식 조합만 가능):
  부상 = if Heart_Rate < 90 and Load > 70: 높음
  
  신경망 (임의의 비선형 함수 근사 가능):
  부상 = f(e^(Heart_Rate) × √Load × sin(Sleep))
  + 은닉층이 여러 개이면 더 복잡한 조합 가능
  ```

- **근사 능력 (유니버설 근사 정리)**:
  ```
  충분한 뉴런이 있는 신경망은 어떤 연속함수도 근사 가능
  
  현재 프로젝트:
  5 → 128 → 64 → 32 → 1
  
  뉴런 수: 약 27,000개 가중치
  → 복잡한 비선형 함수 표현 가능
  ```

### 6.7 프로젝트 MLP 성능 분석
- **성능**:
  ```
  NBA: 52.0% (LGB 68.6%보다 낮음) ✗
  Football: 64.3% (LGB 71.1%보다 낮음) ✗
  Multimodal: 90.4% (LGB 25.9%보다 대폭 높음) ⭐⭐⭐
  ```

- **왜 센서 데이터에만 좋은가?**:
  ```
  Multimodal 특성:
  31개 센서의 복잡한 상호작용
  예: HR ↓ + Load ↑ + Sleep < 70 + 여름 + Rest > 5
  
  트리는 이런 조합을 표현 어려움:
  if (HR < 90 AND Load > 70 AND Sleep < 70 AND Month == 6 AND Rest > 5)
  → 깊이가 매우 깊어짐 (오버피팅)
  
  신경망은 은닉층에서 자동으로 학습:
  은닉층 1: HR × Load 상호작용
  은닉층 2: (HR × Load) × Sleep 3중 상호작용
  은닉층 3: 복잡한 패턴 최종 정제
  → 효율적으로 표현 가능 ✓
  ```

- **장점**:
  - 비선형 상호작용 자동 학습
  - SMOTE와 완벽 시너지
  - 센서 데이터 최적화
  - 특성 엔지니어링 불필요
  
- **단점**:
  - **해석 불가능**: Feature Importance 추출 불가
  - **블랙박스**: "왜 부상 예측했는가?" 설명 불가
  - **소규모 데이터 오버피팅**: 45건(Football)은 위험
  - **의료 신뢰도**: 의료진이 "신경망 알고리즘"을 불신할 가능성
  - **학습 속도**: CPU에서 느림 (GPU 권장)
  - **재현성**: 초기화 난수에 따라 결과 변동

- **코드 예시 (프로젝트의 MLP)**:
  ```python
  from torch.utils.data import DataLoader, TensorDataset
  from imblearn.over_sampling import SMOTE
  
  # SMOTE 적용
  smote = SMOTE(random_state=42)
  X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
  
  # 텐서 변환
  X_train_t = torch.FloatTensor(X_train_res.values)
  X_test_t = torch.FloatTensor(X_test.values)
  y_train_t = torch.FloatTensor(y_train_res.values).reshape(-1, 1)
  y_test_t = torch.FloatTensor(y_test.values).reshape(-1, 1)
  
  # 모델, 손실, 최적화기
  model = InjuryMLP(input_size=X_train.shape[1])
  criterion = nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  # 50 에포크 훈련
  for epoch in range(50):
      optimizer.zero_grad()
      outputs = model(X_train_t)
      loss = criterion(outputs, y_train_t)
      loss.backward()
      optimizer.step()
  
  # 테스트
  with torch.no_grad():
      pred = (model(X_test_t) > 0.5).float()
      recall = recall_score(y_test, pred.numpy())
      print(f"Recall: {recall:.4f}")
  ```

- **성능 향상 아이디어**:
  ```python
  # 1. 은닉층 크기 조정
  # 현재: 128 → 64 → 32
  # 시도: 256 → 128 → 64 (더 큰 모델)
  
  # 2. 에포크 증가
  # 현재: 50 에포크
  # 시도: 200 에포크 (더 깊은 학습)
  
  # 3. 학습률 조정
  # 현재: 0.001
  # 시도: 0.0001 (더 느린, 안정적 수렴)
  
  # 4. 드롭아웃 강화
  # 현재: 30%
  # 시도: 50% (더 강한 정규화, Football 같은 소규모 데이터)
  
  # 5. 배치 정규화 (Batch Normalization) 추가
  # 각 층의 입력을 표준화 → 더 안정적 훈련
  ```
  ```

### 6.4 PyTorch 구현
- **라이브러리**: Python 딥러닝 프레임워크
  
- **주요 컴포넌트**:
  - `torch.nn.Module`: 신경망 기본 클래스
  - `torch.nn.Linear`: 완전연결층
  - `torch.nn.ReLU`: ReLU 활성화
  - `torch.nn.Sigmoid`: Sigmoid 활성화
  - `torch.nn.Dropout`: 드롭아웃
  - `torch.nn.BCELoss`: 이진 교차 엔트로피 손실
  - `torch.optim.Adam`: Adam 최적화기
  
- **텐서(Tensor)**: PyTorch의 데이터 구조
  - NumPy 배열과 유사하지만 GPU 지원
  - `torch.FloatTensor()`: 32비트 부동소수점

### 6.5 드롭아웃 (Dropout)
- **개념**: 학습 중 뉴런을 무작위로 비활성화
  - 30% 드롭아웃 = 70%만 활성화
  
- **효과**:
  - 과적합 방지
  - 앙상블 효과 (여러 네트워크 조합처럼)
  
- **적용 시점**: 은닉층 사이
  - 입력층과 출력층에는 적용 안 함

### 6.6 에포크(Epoch)와 배치(Batch)
- **에포크**: 전체 데이터를 한 번 학습
  - 현재: 50 에포크 = 전체 데이터 50회 학습
  
- **배치**: 한 번에 처리할 샘플 수
  - 배치가 작음 → 더 자주 업데이트, 노이지
  - 배치가 큼 → 안정적, 메모리 사용

### 6.7 프로젝트의 MLP 성능
- **장점**: 비선형 상호작용 자동 학습
  
- **성능**:
  - NBA: 52.0% (LGB 68.6%보다 낮음)
  - Football: 64.3% (LGB 71.1%보다 낮음)
  - **Multimodal: 90.4% ⭐** (LGB 25.9%보다 대폭 높음)
  
- **인사이트**: 센서 데이터의 복잡한 상호작용은 신경망이 효과적!

---

## 7. 데이터셋 특성 (프로젝트의 3가지)

### 7.1 NBA 데이터
- **규모**: 27,105건 (10년)
- **부상율**: 6.5%
- **특성**:
  - 거대한 샘플 → 오버피팅 위험 낮음
  - 역사 데이터 → 패턴 안정적
  - 텍스트 기반 라벨 → 텍스트 전처리 필요
  
- **피처**:
  - Month: 부상 발생 월 (계절성)
  - Days_Missed: 결장 기간
  - Load_Score: 훈련 강도 (대체 피처)
  
- **최적 모델**: LightGBM (68.6%)

### 7.2 Football 데이터
- **규모**: 45건 (Newcastle FC, 1시즌)
- **부상율**: 45.1% (거의 균형)
- **특성**:
  - 매우 작은 샘플 → 오버피팅 위험 높음
  - 시즌별 데이터 → 제한적 패턴
  - 풍부한 메타데이터 (44개 컬럼)
  
- **피처**:
  - Age: 선수 나이
  - FIFA rating: 선수 능력치
  - Month: 부상 발생 월
  - Load_Score: 훈련 강도
  - Match records: 경기 전/중/후 성적
  
- **특수성**: 각 선수의 경기 성적 데이터 (상대팀, 점수, 플레이어 레이팅)

- **최적 모델**: LightGBM (71.1%)

### 7.3 Multimodal (웨어러블 센서) 데이터
- **규모**: 5,430개 샘플
- **부상율**: 5%
- **센서 수**: 31개
- **특성**:
  - 연속 값 시계열 데이터
  - 복잡한 비선형 상호작용
  - 생리적 신호의 미묘한 패턴
  
- **센서 카테고리**:
  - **심혈관**: heart_rate, spo2 (산소포화도)
  - **신경/근육**: emg_amplitude (근전도), fatigue_index
  - **온도**: skin_temp, ambient_temp, heat_index
  - **혈압**: bp_systolic, bp_diastolic
  - **호흡**: respiratory_rate
  - **피부 저항**: gsr (갈바닉 피부 반응)
  - **가속도/동작**: acceleration, angular_velocity, body_orientation
  - **지면 반력**: ground_reaction_force, impact_force
  - **운동 패턴**: step_count, cadence (걸음 빈도), jump_height
  - **ROM**: range_of_motion (관절 가동범위)
  - **대칭성**: gait_symmetry (보행 대칭성)
  - **속도**: speed
  - **환경**: altitude (고도)
  - **습도**: humidity
  - **훈련**: training_duration, workload_intensity, rest_period, repetition_count
  - **이력**: previous_injury_history, acc_rms
  
- **최적 모델**: PyTorch MLP (90.4%)
  - 이유: 센서 간 복잡한 상호작용 (HR ↓ + Load ↑ + Sleep < 70 등)

---

## 8. 모델 비교 및 선택 전략

### 8.1 전체 성능표
| 모델 | NBA | Football | Multimodal | 추천도 |
|------|-----|----------|-----------|--------|
| Random Forest | 64.6% | 67.8% | 33.3% | ⭐⭐ |
| XGBoost | 66.0% | 69.2% | 24.1% | ⭐⭐ |
| LightGBM | 68.6% | 71.1% | 25.9% | ⭐⭐⭐ |
| PyTorch MLP | 52.0% | 64.3% | 90.4% | ⭐⭐⭐ |

### 8.2 선택 기준
- **전통 데이터(구조화, 작은 특성 수)**:
  - LightGBM 선택
  - 이유: 속도, 정확성, 해석성 균형
  
- **센서 데이터(비선형, 많은 특성)**:
  - PyTorch MLP 선택
  - 이유: 복잡한 상호작용 학습 능력
  
- **해석성 중시**:
  - Random Forest 또는 LightGBM
  - Feature Importance 추출 가능
  
- **의료진 신뢰**:
  - Random Forest (가장 설명 가능)
  - 규칙 기반처럼 보임

---

## 9. 프로젝트 워크플로우

### 9.1 단계별 프로세스
1. **데이터 로드**: `pd.read_csv()`
2. **탐색적 데이터 분석 (EDA)**:
   - 결측치 확인
   - 통계 기초 정보
   - 클래스 불균형 확인
3. **데이터 전처리**:
   - 날짜 파싱: `pd.to_datetime()`
   - 텍스트 처리: `str.contains()`
   - 결측치 처리: `fillna()`
4. **특성 엔지니어링**:
   - 월, 요일 추출
   - 피처 선택
5. **데이터 분할**: `train_test_split(stratify=y)`
6. **SMOTE 적용**: 훈련 데이터에만
7. **모델 학습**: 4개 모델 병렬 학습
8. **성능 평가**: Recall 기준
9. **결과 저장**: CSV 파일

### 9.2 코드 흐름
```python
# 1. 데이터 로드
import pandas as pd
df = pd.read_csv('data.csv')

# 2. 전처리
df['Date'] = pd.to_datetime(df['Date'])
df['Injured'] = df['Notes'].str.contains('out', case=False).astype(int)

# 3. 피처 엔지니어링
df['Month'] = df['Date'].dt.month
features = ['Month', 'Load_Score']
X = df[features]
y = df['Injured']

# 4. 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. SMOTE 적용 (훈련 데이터만)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. 모델 학습
from src.models import train_lgb
recall, model = train_lgb(X_train, X_test, y_train, y_test, "NBA")

# 7. 결과 저장
import pandas as pd
results_df = pd.DataFrame({'Model': ['LGB'], 'Recall': [recall]})
results_df.to_csv('results.csv', index=False)
```

---

## 10. 디버깅 및 문제 해결

### 10.1 일반적인 문제
- **메모리 부족**: `n_jobs=-1` 제거, 배치 크기 감소
- **느린 학습**: `n_estimators` 감소, GPU 사용
- **과적합**: 정규화 강화, 데이터 증강
- **과소적합**: 모델 복잡도 증가, 특성 추가
- **불균형 데이터 낮은 성능**: SMOTE 적용, Recall 지표 확인

### 10.2 성능 개선 팁
- Recall이 낮으면: 임계값 낮추기, SMOTE 강도 조정
- Precision이 낮으면: 임계값 높이기, 특성 추가
- 과적합이면: 정규화 강화, 하이퍼파라미터 조정
- 과소적합이면: 모델 복잡도 증가, 특성 공학

---

## 11. 고급 주제

### 11.1 앙상블(Ensemble)
- **개념**: 여러 모델 조합으로 더 강력한 모델 생성
  
- **종류**:
  - **Voting**: 다수결 투표 (분류) 또는 평균 (회귀)
  - **Stacking**: 메타 모델이 기본 모델의 출력 학습
  - **Blending**: Stacking의 단순 버전
  
- **예시**:
  ```python
  # NBA: LGB (68.6%) + MLP (52.0%) 결합
  # 동적 가중치: LGB 70%, MLP 30%
  # → 예상 성능: 약 65%?
  ```

### 11.2 전이 학습(Transfer Learning)
- **개념**: 큰 데이터셋으로 학습한 모델을 작은 데이터셋에 재사용
  
- **장점**: 소규모 데이터에서도 강력한 성능
  
- **응용**: Football(45건) 같은 소규모 데이터에 효과적

### 11.3 설명 가능한 AI (Explainable AI)
- **SHAP (SHapley Additive exPlanations)**:
  - 각 특성이 개별 예측에 기여하는 정도
  - Feature Importance보다 상세함
  
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - 특정 예측을 설명하는 지역적 근사 모델
  
- **의료 적용**: "왜 이 선수에게 부상 위험 경고했는가?" 설명 가능

### 11.4 강건성(Robustness) 테스트
- **개념**: 입력 작은 변화에 모델이 안정적인가?
  
- **방법**:
  - 노이즈 추가
  - 피처 섭동(perturbation)
  - 적대적 예시(adversarial examples)
  
- **의료**: 센서 오류에도 안정적인 예측 필요

---

## 12. 프로덕션 배포

### 12.1 모델 저장 및 로드
- **Pickle**: Python 객체 직렬화
  ```python
  import pickle
  with open('model.pkl', 'wb') as f:
      pickle.dump(model, f)
  ```
  
- **Joblib**: 큰 모델 최적화
  ```python
  from joblib import dump, load
  dump(model, 'model.pkl')
  ```

### 12.2 REST API 생성
- **Flask**: 경량 웹 프레임워크
  ```python
  from flask import Flask, request, jsonify
  app = Flask(__name__)
  
  @app.route('/predict', methods=['POST'])
  def predict():
      data = request.json
      prediction = model.predict(data)
      return jsonify({'injury_risk': float(prediction)})
  ```

### 12.3 Docker 배포
- **개념**: 일관된 환경에서 모델 실행
  
- **Dockerfile**:
  ```dockerfile
  FROM python:3.8
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["python", "app.py"]
  ```

### 12.4 모니터링
- **성능 드리프트(Performance Drift)**: 시간 경과에 따른 성능 저하
- **데이터 드리프트(Data Drift)**: 입력 데이터 분포 변화
- **모델 재학습**: 주기적 업데이트 필요

---

## 13. 시각화 및 보고

### 13.1 Matplotlib/Seaborn 기본
- **라인 그래프**: 시계열 데이터
- **막대 그래프**: 범주형 비교
- **히스토그램**: 분포
- **산점도**: 관계성
- **히트맵**: 상관계수, 혼동행렬
- **박스플롯**: 이상치 탐지

### 13.2 프로젝트 시각화
- **모델 성능 비교 차트**: 막대 그래프 (Recall 비교)
- **Feature Importance**: 수평 막대 그래프
- **고위험군 시각화**: 산점도 (2D 센서 데이터)

---

## 14. 학습 순서 제안

### 초급 (1-2주)
1. Python 기초
2. Pandas, NumPy 데이터 처리
3. 데이터 전처리 (결측치, 정규화)
4. 머신러닝 기초 개념

### 중급 (3-4주)
5. 모델 평가 지표 (Accuracy, Precision, Recall)
6. 클래스 불균형 (SMOTE)
7. 의사결정 트리, Random Forest
8. XGBoost, LightGBM
9. 하이퍼파라미터 튜닝

### 고급 (5-6주)
10. 신경망 기초 (퍼셉트론)
11. PyTorch MLP 구현
12. 역전파 알고리즘
13. 드롭아웃, 정규화
14. 프로덕션 배포 (Flask, Docker)

---

## 15. 참고 자료

### 15.1 온라인 강좌
- **Kaggle**: 실전 문제 풀이
- **Coursera**: Andrew Ng 머신러닝 강좌
- **Fast.ai**: 실전 중심 딥러닝
- **YouTube**: StatQuest with Josh Starmer (트리 모델 직관)

### 15.2 책
- **패턴 인식과 머신러닝** (Bishop)
- **파이썬을 이용한 머신러닝** (Müller & Guido)
- **딥러닝** (Goodfellow et al.)

### 15.3 공식 문서
- pandas: https://pandas.pydata.org/docs/
- scikit-learn: https://scikit-learn.org/stable/documentation.html
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- PyTorch: https://pytorch.org/docs/stable/

### 15.4 의료 AI 관련
- **의료 데이터 윤리**: 개인정보, 동의서
- **규제 준수**: FDA 승인 프로세스
- **임상 검증**: 다기관 연구

---

## 16. 실습 과제

### 16.1 초급
- [ ] 각 데이터셋의 기본 통계 계산
- [ ] 결측치 비율 확인
- [ ] 클래스 불균형 시각화

### 16.2 중급
- [ ] SMOTE 적용 전후 성능 비교
- [ ] 4개 모델 학습 및 Recall 비교
- [ ] Feature Importance 해석

### 16.3 고급
- [ ] Football 데이터에 전이 학습 적용
- [ ] 특정 선수의 부상 위험도 예측
- [ ] Flask API로 실시간 예측 시스템 구축

---

이 로드맵을 따라 체계적으로 학습하면 프로젝트의 모든 요소를 완벽히 이해할 수 있습니다!
