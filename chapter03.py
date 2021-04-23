import numpy as np
import pandas as pd
from pandas.core.indexing import is_label_like
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# 3.2 観測されデータの概念を調べる
d = pd.read_csv("./data/data3a.csv")
print(d)
print(d.describe())
print(d.f.value_counts())

# 3.3 統計もリングの前にデータを図示する

# # 図3.2の描画
# C = d[d.f == 'C']
# T = d[d.f == 'T']
# # sns.scatterplot(x=C.x, y=C.y, lab)
# sns.scatterplot(x='x', y='y', hue='f', data=d)
# # plt.legend()
# plt.show()

# # 図3.3の描画
# sns.boxplot(x=d.f, y=d.y, data=d)
# plt.show()

# 3.4.1 線形予測子と対数リンク関数

# # 図3.4の描画
# xi = np.arange(-4, 5, 0.05)
# b1 = -2
# b2 = -0.8
# lam = np.exp(b1 + (b2 * xi))
# plt.plot(xi, lam, label="$\{ \\beta, \\beta_2 \} = \{-2, -0.8 \}$")
# b1 = -1
# b2 = 0.4
# lam = np.exp(b1 + (b2 * xi))
# plt.plot(xi, lam, label="$\{ \\beta, \\beta_2 \} = \{-1, 0.4 \}$")
# plt.axvline(0, ymax=1, ymin=0, ls=':')
# plt.legend()
# plt.show()

# 3.4.2 あてはめと当てはまりのよさ

import statsmodels.api as sm
import statsmodels.formula.api as smf

# statsmodelsによるglm
formula = "y ~ x"
model = smf.glm(data=d,
                formula=formula,
                family=sm.families.Poisson())
fit = model.fit()
# 要約の出力
print(fit.summary())

# 最大対数尤度を確認(log-likelihood)
print(fit.llf)
# 自由度の確認(ただし切片は数に含めない)
fit.df_model

# 図3.7の描画
xx = np.linspace(d.x.min(), d.x.max(), 100)
lam = np.exp(fit.params['Intercept'] + fit.params['x'] * xx)
sns.scatterplot(x='x', y='y', hue='f', data=d)
sns.lineplot(x=xx, y=lam)
plt.show()

# 3.5 説明変数が因子型の統計モデル
formula = "y ~ f"
model = smf.glm(data=d,
                formula=formula,
                family=sm.families.Poisson())
fit_f = model.fit()

# 要約の出力
fit_f.summary()
# 最大対数尤度を確認
fit_f.llf
# 自由度（ただし切片は含めない）
fit_f.df_model

# 3.6 説明変数が数量型+因子型の統計モデル
formula = "y ~ x + f"
model = smf.glm(data=d,
                formula=formula,
                family=sm.families.Poisson())
fit_all = model.fit()

# 要約の出力
print(fit_all.summary())
# 最大対数尤度
print(fit_all.llf)
# 自由度（ただし切片は含めない）
print(fit_all.df_model)