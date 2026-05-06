# train_eval_fixed.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
import joblib
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs(r'C:\Паша\Проекты\VKR', exist_ok=True)

# 0. Загрузка и кодирование меток
X_full = np.load(r'C:\\Паша\\Проекты\\VKR\\X_train_smote.npy')
y_full = np.load(r'C:\\Паша\\Проекты\\VKR\\y_train_smote.npy', allow_pickle=True)

le = LabelEncoder()
y_full_int = le.fit_transform(y_full)

X_train, _, y_train_int, _ = train_test_split(
    X_full, y_full_int, train_size=500_000, random_state=42, stratify=y_full_int
)

X_test = np.load(r'C:\\Паша\\Проекты\\VKR\\X_test.npy')
y_test = np.load(r'C:\\Паша\\Проекты\\VKR\\y_test.npy', allow_pickle=True)
y_test_int = le.transform(y_test)

print(f'Тренировочная: {X_train.shape}  Тестовая: {X_test.shape}')
print('Классы (int):', np.unique(y_test_int))

# 1. Random Forest
print('\n=== Random Forest ===')
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                            n_jobs=-1, random_state=42, oob_score=True)
rf.fit(X_train, y_train_int)
joblib.dump(rf, r'C:\\Паша\\Проекты\\VKR\\rf_model.pkl')
print('OOB-score:', rf.oob_score_)

# 2. XGBoost (без early stopping — для совместимости с XGBoost 3.1.2)
print('\n=== XGBoost ===')
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train_int, eval_set=[(X_test, y_test_int)], verbose=False)
joblib.dump(xgb, r'C:\\Паша\\Проекты\\VKR\\xgb_model.pkl')

# 3. Метрики (все на числовых метках!)
results = []
for name, model in [('Random Forest', rf), ('XGBoost', xgb)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_int, y_pred)
    macro_f1 = classification_report(y_test_int, y_pred, output_dict=True, zero_division=0)['macro avg']['f1-score']
    roc_auc = roc_auc_score(y_test_int, model.predict_proba(X_test), multi_class='ovr', average='macro')
    results.append({'Model': name, 'Accuracy': acc, 'Macro-F1': macro_f1, 'ROC-AUC': roc_auc})
    print(f'\n{name}  Accuracy={acc:.3f}  Macro-F1={macro_f1:.3f}  ROC-AUC={roc_auc:.3f}')

df_res = pd.DataFrame(results)
print('\nСводная таблица:')
print(df_res.round(3))
df_res.to_csv(r'C:\\Паша\\Проекты\\VKR\\classification_results.csv', index=False)

# 4. ROC-AUC (micro-average) — на числовых метках
classes = np.unique(y_test_int)
y_test_bin = label_binarize(y_test_int, classes=classes)

fpr_rf, tpr_rf, _ = roc_curve(y_test_bin.ravel(), rf.predict_proba(X_test).ravel())
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_bin.ravel(), xgb.predict_proba(X_test).ravel())

plt.figure(figsize=(6, 5))
plt.plot(fpr_rf, tpr_rf, lw=2, linestyle='--',
         label=f'Random Forest (micro-ROC-AUC = {auc(fpr_rf, tpr_rf):.3f})')
plt.plot(fpr_xgb, tpr_xgb, lw=2, linestyle='-',
         label=f'XGBoost (micro-ROC-AUC = {auc(fpr_xgb, tpr_xgb):.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC-AUC (micro-average)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(r'C:\\Паша\\Проекты\\VKR\\roc_curves.png', dpi=300)
plt.show()

print('\nГотово! Модели и отчёты в папке VKR')