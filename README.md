# Sport-Specific Injury Prediction Cross-Sport Meta Insights

이 저장소는 세 가지 스포츠 데이터셋을 대상으로, `Recall`을 우선하는 부상 의사결정 지원 실험을 다룹니다.

- `NBA`: 부상 리포트 텍스트와 일정 맥락 데이터
- `Football`: 부상 전 경기 맥락이 포함된 선수 부상 영향 데이터
- `Multimodal`: 웨어러블 유사 생리·운동역학 센서 데이터

프로젝트의 핵심 목적은 그대로 유지됩니다. 각 데이터셋에서 여러 모델 계열을 비교하고, 놓치는 부상을 줄이기 위해 `Recall`을 우선하며, 기존 스크립트 진입점도 계속 실행 가능하도록 유지합니다.

## 변경 사항

코드베이스는 재현성, 유지보수성, 분석 신뢰도를 높이기 위해 다음과 같이 리팩터링되었습니다.

- 기존의 랜덤 `Load_Score` 생성 방식을 제거하고 결정론적 피처 엔지니어링으로 교체
- 데이터 로딩과 스키마 검증을 `src/data/`로 분리
- 피처 엔지니어링 로직을 `src/features/`로 분리
- CV 안전 학습 파이프라인과 표준화된 평가 지표를 `src/train/`으로 정리
- 설명 가능성 관련 placeholder를 `src/xai/`로 정리
- 데이터셋별 실험 설정을 `configs/*.yaml`에 분리
- 인사이트 시각화와 `results/MODEL_COMPARISON.md`를 하드코딩 대신 표준 산출물 기반으로 재생성하도록 변경

## 저장소 구조

```text
configs/                  데이터셋별 YAML 설정 파일
data/                     원본 CSV 데이터셋
insights/                 생성된 시각화와 시각화 진입점
notebooks/                하위 호환용 실행 스크립트
results/                  표준화된 결과 테이블과 생성 보고서
src/data/                 로더와 스키마 검증
src/features/             결정론적 피처 빌더와 정규화 유틸리티
src/train/                설정 파싱, 파이프라인, 지표, 리포팅, CLI
src/xai/                  SHAP/PDP placeholder
tests/                    스모크 테스트
```

## 데이터셋 메모

각 설정 파일에는 데이터 경로, 라벨 생성 규칙, 피처 목록, 모델 설정, 불균형 처리 방식, 시드가 정리되어 있습니다.

| 데이터셋 | 원본 파일 | 라벨 규칙 | 현재 분석 의도 |
| --- | --- | --- | --- |
| NBA | `data/injuries_2010-2020.csv` | `Notes`에서 키워드 기반 타깃 생성 | 부상 리포트 텍스트 기반 부상 플래그 프록시 |
| Football | `data/player_injuries_impact.csv` | 날짜 정리 후 `Days_Out >= 30` | 심각한 부상 기간 프록시 |
| Multimodal | `data/sports_multimodal_data.csv` | 기존 `injury_risk` 컬럼 사용 | 이진 위험 분류 |

## 재현 가능한 실행 방법

의존성 설치:

```bash
python -m pip install -r requirements.txt
```

새 진입점으로 단일 데이터셋 실행:

```bash
python -m src.train.run --config configs/nba.yaml
python -m src.train.run --config configs/football.yaml
python -m src.train.run --config configs/multimodal.yaml
```

기존 notebook 스타일 명령도 그대로 동작하며 동일한 파이프라인을 호출합니다.

```bash
python notebooks/NBA.py
python notebooks/Football.py
python notebooks/Multimodal.py
```

최신 표준 산출물 기준으로 시각화와 비교 markdown 재생성:

```bash
python insights/visualization.py
```

스모크 테스트 실행:

```bash
python -m unittest discover -s tests -v
```

## 표준 산출물

각 실행은 다음 파일들을 생성합니다.

- `results/<dataset>_results.csv`: 모델별 recall, precision, F1, PR-AUC, threshold, confusion matrix 카운트
- `results/artifacts/<dataset>/models/`: 직렬화된 모델 아티팩트
- `results/artifacts/<dataset>/predictions/`: 시각화와 감사용 모델별 예측 테이블
- `results/artifacts/<dataset>/run_metadata.json`: 실행 설정, 노트, 피처 목록, 라벨 비율
- `results/MODEL_COMPARISON.md`: 최신 크로스 데이터셋 비교 결과
- `insights/*.png`: 최신 아티팩트 기준으로 다시 생성된 시각화

트리 기반 모델은 `joblib`으로 저장되며, PyTorch MLP는 `torch.save`로 저장됩니다.

## 분석 신뢰성 메모

이번 리팩터링은 기존 코드의 몇 가지 문제를 의도적으로 보완합니다.

- 과거 `README.md`, `results/MODEL_COMPARISON.md`, `insights/visualization.py`에 적힌 Football 및 Multimodal 서술형 지표가 `results/`의 CSV 산출물과 일치하지 않았음
- 과거 NBA와 Football 스크립트가 `np.random.uniform(...)`으로 `Load_Score`를 생성해 재실행 시 결과가 달라졌음
- 과거 시각화 코드가 결과 파일을 읽지 않고 recall 값을 하드코딩했음
- 실제 Football CSV는 예전 서술보다 훨씬 크며, 음수 부상 기간 같은 잘못된 행이 있어 현재는 피처 생성 과정에서 제거됨

다만 아래 한계는 여전히 남아 있습니다.

- NBA 라벨은 원본 파일에 더 깔끔한 이진 타깃이 없어 부상 리포트 텍스트 기반 프록시 라벨을 사용함
- Football은 부상 이벤트 데이터 중심이라 완전한 무부상 노출 이력을 반영한 태스크가 아니라 심각도 프록시 성격이 남아 있음
- SHAP와 Partial Dependence 출력은 실제 구현 전까지 placeholder 상태임

## 권장 작업 흐름

1. `python -m src.train.run --config ...`로 하나 이상의 데이터셋 설정을 실행합니다.
2. `results/<dataset>_results.csv`와 `results/artifacts/<dataset>/run_metadata.json`을 확인합니다.
3. 필요하면 `python insights/visualization.py`로 시각화를 다시 생성합니다.
4. `results/MODEL_COMPARISON.md`에서 현재 기준의 크로스 데이터셋 요약을 검토합니다.

## 재현성 기본 원칙

- 전역 시드는 설정 파일 단위로 관리됩니다.
- train/test split은 stratified 방식으로 수행됩니다.
- SMOTE는 `imblearn` 파이프라인 내부에서 적용되어 리샘플링이 학습 경로 안에만 머무릅니다.
- threshold 선택은 중앙화되어 있으며 설정으로 제어할 수 있습니다.
- 필요 시 stratified cross-validation을 설정에서 활성화할 수 있습니다.

## 다음 확장 방향

- `src/xai/` 아래에 완전한 SHAP 및 PDP 생성 기능 구현
- 더 엄격한 recall 목표가 필요할 때 calibration 기반 threshold 선택 추가
- 더 정교한 workload 데이터가 확보되면 도메인 피처 확장
