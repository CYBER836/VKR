import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np 

df = pd.read_csv(r'C:\Паша\Проекты\VKR\cicids2017_clean.csv')

# 1.1 выделяем X и y

X = df.drop(columns=['Label'])
y = df['Label']

# убираем inf/NaN
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Label'] = y.values

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Label', alpha=0.6, s=8, legend=False)
plt.title('PCA: 16-D → 2-D (объяснённая дисперсия {:.1f} %)'.format(
    pca.explained_variance_ratio_.sum()*100))
plt.xlabel('PC1 ({:.1f} %)'.format(pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC2 ({:.1f} %)'.format(pca.explained_variance_ratio_[1]*100))
plt.tight_layout()
plt.savefig('pca_2d.png', dpi=300)
plt.show()