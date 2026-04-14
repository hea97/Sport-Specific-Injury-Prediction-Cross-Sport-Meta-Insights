# Model Comparison

Generated from the latest standardized training artifacts.

| Dataset | Model | Recall | Precision | F1 | PR-AUC | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Football | MLP | 0.5577 | 0.5800 | 0.5686 | 0.6488 | 0.50 |
| Football | XGB | 0.5000 | 0.5652 | 0.5306 | 0.5285 | 0.50 |
| Football | RF | 0.4615 | 0.5714 | 0.5106 | 0.5869 | 0.50 |
| Football | LGB | 0.4423 | 0.5476 | 0.4894 | 0.5353 | 0.50 |
| Multimodal | MLP | 0.4444 | 0.1364 | 0.2087 | 0.1739 | 0.50 |
| Multimodal | XGB | 0.3519 | 0.1810 | 0.2390 | 0.1621 | 0.50 |
| Multimodal | RF | 0.3333 | 0.2308 | 0.2727 | 0.2050 | 0.50 |
| Multimodal | LGB | 0.3148 | 0.1667 | 0.2179 | 0.1648 | 0.50 |
| NBA | RF | 0.9829 | 0.9322 | 0.9569 | 0.9861 | 0.50 |
| NBA | XGB | 0.9771 | 0.9688 | 0.9730 | 0.9970 | 0.50 |
| NBA | LGB | 0.9714 | 0.9798 | 0.9756 | 0.9974 | 0.50 |
| NBA | MLP | 0.9457 | 0.8665 | 0.9044 | 0.9608 | 0.50 |

## Best Current Model Per Dataset

- **Football**: MLP (Recall 0.5577, PR-AUC 0.6488)
- **Multimodal**: MLP (Recall 0.4444, PR-AUC 0.1739)
- **NBA**: RF (Recall 0.9829, PR-AUC 0.9861)
