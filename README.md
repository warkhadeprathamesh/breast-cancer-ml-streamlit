# Breast Cancer Diagnosis: Classification Models with Streamlit

M.Tech (DSE), BITS Pilani WILP · Machine Learning · Assignment 2

**Live app:** https://<your-app-name>.streamlit.app
**Repository:** https://github.com/<your-username>/breast-cancer-ml-streamlit

## a. Problem Statement
Classify a breast mass as **malignant** or **benign** from 30 measurements of cell nuclei taken from a fine-needle aspirate image. This is a **binary classification** problem. Five classical models (Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, SVM) are trained on the same data and split, compared on six metrics, and exposed through an interactive Streamlit app. Benign (1) is the positive class, so the costlier clinical error (a malignant tumour predicted benign) shows up as a *false positive* in the confusion matrix; precision and MCC are therefore read together with accuracy.

## b. Dataset Description
| Attribute | Details |
|---|---|
| Source | UCI ML Repository, Breast Cancer Wisconsin (Diagnostic) |
| Instances | 569 (≥ 500 required) |
| Features | 30 numeric (≥ 12 required): mean, standard error and worst value of 10 nuclei properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) |
| Target | `1` = Benign (357, 62.7%), `0` = Malignant (212, 37.3%) |
| Missing values | None |
| Split | Stratified 80/20 → 455 train / 114 test, `random_state=42` |
| Preprocessing | `StandardScaler` inside a `Pipeline` (fit on train only) for Logistic Regression, kNN and SVM |
| Hyper-parameters | LogReg `max_iter=1000`; Decision Tree `max_depth=5`; kNN `k=7`; GaussianNB defaults; SVM RBF, `C=1.0`, `gamma='scale'` |

## c. GitHub Repository
```
breast-cancer-ml-streamlit/
├── app.py                  # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv           # 114 held-out rows (30 features + target) for the app
└── model/
    ├── ml_assignment2.ipynb   # EDA, training, evaluation (run on BITS Virtual Lab)
    ├── train_models.py        # reproducible training script
    ├── metrics.csv            # final comparison table
    └── *.pkl                  # 5 saved models
```
Reproduce: `pip install -r requirements.txt && python model/train_models.py && streamlit run app.py`

## d. Models Used

### Comparison table (test set, n = 114)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Top performer: 98.25% accuracy, AUC 0.9954, MCC 0.9623, only 2 errors (1 FP, 1 FN). Train accuracy (98.9%) is close to test accuracy, so it generalises well. After scaling, the classes are nearly linearly separable, which suits a linear model; it is also fast and interpretable. |
| Decision Tree | Weakest: accuracy 92.11%, AUC 0.9163, MCC 0.8341. Train 99.3% vs test 92.1% shows overfitting even at `max_depth=5`. It labelled 6 benign cases malignant and 3 malignant cases benign. About 71% of its importance comes from one feature (worst radius). |
| kNN | Strong: accuracy 97.37%, AUC 0.9884, recall 1.000 (no benign case flagged malignant). However, 3 malignant cases were predicted benign, lowering precision to 0.96. Very sensitive to feature scaling and to k. |
| Naive Bayes | Fast but below the leaders: accuracy 93.86%, MCC 0.8676. High AUC (0.9878) means probabilities rank well, yet hard predictions suffer (4 malignant→benign, 3 benign→malignant). The independence assumption is violated by strongly correlated features (radius, perimeter, area). |
| Support Vector Machine | Ties with Logistic Regression: 98.25% accuracy, AUC 0.9950, MCC 0.9623, 2 errors. The RBF kernel on scaled data gives a smooth boundary with little overfitting (train 98.2%). Harder to tune (C, gamma) and interpret. |
| **Overall winner** | **Logistic Regression** (SVM is a statistical tie): equal best accuracy/F1/MCC, highest AUC, and it is simpler, faster and more interpretable. It and SVM also missed only 1 malignant tumour, versus 3 for kNN and the Decision Tree and 4 for Naive Bayes. Caveat: with 114 test rows, one prediction moves accuracy by 0.9 points. |

## Streamlit App Features
- **CSV upload** (test data) with preview and column validation
- **Model dropdown** with all five models
- **Metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** heatmap and **classification report**
- Extras: ROC curve, adjustable decision threshold, all-model comparison table and chart, sample CSV download
