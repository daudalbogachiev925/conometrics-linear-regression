"""МНК: оценка, диагностика, прогноз."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200

# Данные: зависимость зарплаты от опыта и образования
experience = np.random.uniform(0, 20, n)
education = np.random.randint(1, 4, n)
salary = 30 + 2.5 * experience + 5 * education + np.random.normal(0, 3, n)

df = pd.DataFrame({
    "salary": salary,
    "experience": experience,
    "education": education
})

# === 1. МНК ===
X = sm.add_constant(df[["experience", "education"]])
y = df["salary"]

model = sm.OLS(y, X).fit()
print(model.summary())

# === 2. Коэффициенты ===
print("\nКоэффициенты:")
print(f"  const:      {model.params['const']:.4f}")
print(f"  experience: {model.params['experience']:.4f}")
print(f"  education:  {model.params['education']:.4f}")

# === 3. Метрики ===
print(f"\nR²:        {model.rsquared:.4f}")
print(f"Adj. R²:   {model.rsquared_adj:.4f}")
print(f"F-stat:    {model.fvalue:.4f}")
print(f"p-value F: {model.f_pvalue:.4e}")

# === 4. Проверка гетероскедастичности ===
bp_test = het_breuschpagan(model.resid, X)
print(f"\nBreusch-Pagan p-value: {bp_test[1]:.4f}")
print("Гетероскедастичность:", "есть" if bp_test[1] < 0.05 else "нет")

# === 5. Визуализация остатков ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(model.fittedvalues, model.resid, alpha=0.6)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Предсказанные значения")
axes[0].set_ylabel("Остатки")
axes[0].set_title("Остатки vs Предсказания")

axes[1].hist(model.resid, bins=20, edgecolor="black")
axes[1].set_title("Распределение остатков")
plt.tight_layout()
plt.savefig("residuals.png")
plt.show()

# === 6. Прогноз ===
new_data = pd.DataFrame({
    "const": [1, 1],
    "experience": [5, 15],
    "education": [2, 3]
})
forecast = model.predict(new_data)
print(f"\nПрогноз зарплаты:\n{forecast}")
