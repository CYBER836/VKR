# feature_importance.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# загружаем модели и энкодер
rf = joblib.load(r'C:\Паша\Проекты\VKR\rf_model.pkl')
xgb = joblib.load(r'C:\Паша\Проекты\VKR\xgb_model.pkl')

# восстановим имена признаков (из train_eval.py)
feature_names = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Average Packet Size',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count'
]

# важности
rf_imp = rf.feature_importances_
xgb_imp = xgb.feature_importances_

# строим график
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest
top_rf = pd.Series(rf_imp, index=feature_names).nlargest(10)
top_rf.plot.barh(ax=ax[0])
ax[0].set_title('Random Forest – TOP-10 признаков')
ax[0].set_xlabel('Mean Decrease Impurity')

# XGBoost
top_xgb = pd.Series(xgb_imp, index=feature_names).nlargest(10)
top_xgb.plot.barh(ax=ax[1])
ax[1].set_title('XGBoost – TOP-10 признаков')
ax[1].set_xlabel('Gain')

plt.tight_layout()
plt.savefig(r'C:\Паша\Проекты\VKR\feature_importance.png', dpi=300)
plt.show()