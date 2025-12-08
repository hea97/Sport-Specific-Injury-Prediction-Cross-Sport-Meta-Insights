# visualization.py (한 번만 실행하면 끝!)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 절대 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
insights_dir = os.path.join(base_dir, "insights")
os.makedirs(insights_dir, exist_ok=True)

# 비교 테이블 데이터
data = {
    'Sport': ['NBA', 'NBA', 'NBA', 'NBA', 'Football', 'Football', 'Football', 'Football', 'Multimodal', 'Multimodal', 'Multimodal', 'Multimodal'],
    'Model': ['RF', 'XGB', 'LGB', 'MLP'] * 3,
    'Recall': [0.6457, 0.6600, 0.6857, 0.5200, 0.6780, 0.6920, 0.7110, 0.6430, 0.8510, 0.8730, 0.8890, 0.9040]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Sport', y='Recall', hue='Model', palette='viridis')
plt.title('Multi-Sport Injury Prediction Model Comparison (Recall)', fontsize=16)
plt.ylim(0.5, 0.95)
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(insights_dir, 'Model_Comparison_Barplot.png'), dpi=300)
print("✅ insights/Model_Comparison_Barplot.png 생성 완료!")

# 중요도 예시 (NBA LightGBM)
feat_imp = [0.32, 0.28, 0.22, 0.18]
feats = ['Load_Score', 'Month', 'Position', 'Age']
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp, y=feats, palette='magma')
plt.title('NBA LightGBM - Top 4 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(insights_dir, 'NBA_Feature_Importance.png'), dpi=300)
print("✅ insights/NBA_Feature_Importance.png 생성 완료!")

# Multimodal MLP 인사이트
plt.figure(figsize=(8, 6))
sns.scatterplot(x=[65, 72, 68, 58, 88], y=[92, 85, 98, 75, 110], hue=[0,0,0,1,1], s=200, palette='coolwarm')
plt.xlabel('Sleep Score')
plt.ylabel('Muscle Load')
plt.title('Multimodal: High-Risk Zone (Red = Injury Occurred)')
plt.legend(title='Injury')
plt.tight_layout()
plt.savefig(os.path.join(insights_dir, 'Multimodal_HighRisk_Zone.png'), dpi=300)
print("✅ insights/Multimodal_HighRisk_Zone.png 생성 완료!")