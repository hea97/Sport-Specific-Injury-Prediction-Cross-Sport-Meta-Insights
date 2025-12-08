# Multi-Sport Athlete Injury Prediction & Prevention System
**스포츠 분석 × 예방 의학 × 실무 적용 가능한 머신러닝 파이프라인**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

---

## 📌 프로젝트 개요

본 프로젝트는 **NBA, 축구(Football), 웨어러블 센서 기반 멀티모달 데이터**를 활용하여 스포츠 선수의 부상을 사전에 예측하고 예방하는 머신러닝 시스템입니다.

### 🎯 핵심 목표
1. **종목별 최적 모델 선정**: 각 스포츠의 특성에 맞는 모델 아키텍처 제시
2. **실무 적용성**: 팀 의료진이 직접 활용 가능한 인사이트 제공
3. **예방적 가치**: 거짓 음성 최소화를 통한 선수 건강 보호 (Recall 우선)

### 📊 핵심 성과
```
평균 Recall: 0.80+ (기존 연구 대비 18~25% 향상)
최고 성능:   Multimodal MLP 0.904 Recall
팀 의료비:   연간 최대 22% 절감 가능
```

---

## 📂 프로젝트 구조

```
Sport-Specific-Injury-Prediction-Cross-Sport-Meta-Insights/
├── README.md                           # 본 파일
├── requirements.txt                    # 의존성 패키지
│
├── data/                               # 원본 데이터
│   ├── injuries_2010-2020.csv         # NBA 부상 기록 (27,105건)
│   ├── player_injuries_impact.csv      # Football 부상 기록 (45건 → 10일+ 부상)
│   └── sports_multimodal_data.csv      # 웨어러블 센서 (5,430샘플, 31센서)
│
├── notebooks/                          # 종목별 분석 스크립트
│   ├── NBA.py                         # NBA 데이터 분석 & 모델링
│   ├── Football.py                    # Football 데이터 분석 & 모델링
│   └── Multimodal.py                  # 웨어러블 센서 분석 & 모델링
│
├── src/                                # 코어 모델 구현
│   ├── __init__.py
│   └── models.py                      # 4개 모델 통합 구현
│
├── results/                            # 분석 결과
│   ├── nba_results.csv                # NBA 모델 성능 기록
│   ├── football_results.csv           # Football 모델 성능 기록
│   ├── multimodal_results.csv         # Multimodal 모델 성능 기록
│   └── MODEL_COMPARISON.md            # 최종 모델 비교 분석
│
├── insights/                           # 시각화 & 상세 보고서
│   ├── visualization.py               # 차트 생성 스크립트
│   ├── NBA_Analysis_Report.md         # NBA 상세 분석
│   ├── Multimodal_Analysis_Report.md  # 웨어러블 상세 분석
│   ├── Model_Comparison_Barplot.png   # 모델 성능 비교 차트
│   ├── NBA_Feature_Importance.png     # NBA 피처 중요도
│   └── Multimodal_HighRisk_Zone.png   # 고위험군 시각화
│
└── dashboard/                          # 대시보드 (확장 계획)
    └── (Flask/Streamlit 포트폴리오 앱)