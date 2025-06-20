import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("ml_6.csv")
print("Пропуски по столбцам:\n", df.isna().sum())

df = df.dropna()

# Определяем целевой столбец и признаки
target = "История пожаров"
# Удаляем из X лишние столбцы, если они есть
X = df.drop(columns=[target, "true", "pred"], errors="ignore")
y = df[target]

# Делим признаки на категориальные и числовые
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Масштабируем числовые признаки
scaler = StandardScaler()
X_scaled_num = pd.DataFrame(scaler.fit_transform(X[num_features]), columns=num_features)

# Объединяем масштабированные числовые и оригинальные категориальные
X_processed = pd.concat(
    [X_scaled_num.reset_index(drop=True), X[cat_features].reset_index(drop=True)],
    axis=1
)
# Считаем индексы категориальных признаков для CatBoost
cat_feature_indices = [X_processed.columns.get_loc(col) for col in cat_features]

# Разбиваем выборку на train и validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Размеры: X_train={X_train.shape}, X_val={X_val.shape}")

# Обучение модели с валидацией
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_feature_indices,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=100,
    random_seed=42
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# Предсказываем метки и вероятности "Да"
y_val_pred = model.predict(X_val)
proba_val = model.predict_proba(X_val)
idx_yes = list(model.classes_).index("Да")
y_val_proba_yes = proba_val[:, idx_yes]

# Выводим точность и отчет
acc = accuracy_score(y_val, y_val_pred)
print("\n--- Метрики на валидационной выборке ---")
print(f"Accuracy: {acc:.4f}")
print("Classification report:")
print(classification_report(y_val, y_val_pred, digits=4))

# Матрица неточностей
cm = confusion_matrix(y_val, y_val_pred, labels=model.classes_)
print("Confusion matrix (строки — истинные, столбцы — предсказанные):")
print(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))

# ROC‑кривая и AUC
fpr, tpr, thresholds = roc_curve(
    (y_val == "Да").astype(int),
    y_val_proba_yes
)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC‑кривая на валидационной выборке")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

importances = model.get_feature_importance(prettified=True)

# Выводим названия колонок и первые строки
print("Колонки в DataFrame с важностями:", importances.columns.tolist())
print(importances.head())

cols = importances.columns.tolist()

if len(cols) == 3:
    feat_col = cols[1]
    imp_col = cols[2]
elif len(cols) == 2:
    feat_col = cols[0]
    imp_col = cols[1]
else:
    feat_col = cols[0]
    imp_col = cols[-1]

plt.figure(figsize=(8, 6))
plt.barh(importances[feat_col], importances[imp_col])
plt.xlabel("Важность")
plt.title("Feature Importance (CatBoost)")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()

# Сохраняем модель CatBoost в формате .cbm
model.save_model("fire_model.cbm")

# Сохраняем масштабировщик
joblib.dump(scaler, "scaler.pkl")

# Сохраняем информацию о признаках
feature_info = {
    "X_columns": X_processed.columns.tolist(),
    "cat_features": cat_features,
    "num_features": num_features
}
joblib.dump(feature_info, "feature_info.pkl")

print("Модель и все артефакты сохранены в файлы: fire_model.cbm, scaler.pkl, feature_info.pkl")

# Загружаем модель CatBoost
from catboost import CatBoostClassifier as _CBC
inference_model = _CBC()
inference_model.load_model("fire_model.cbm")

inference_scaler = joblib.load("scaler.pkl")
info = joblib.load("feature_info.pkl")
X_columns_inf = info["X_columns"]
cat_features_inf = info["cat_features"]
num_features_inf = info["num_features"]

# Функция "чтобы правильно вводились значения пользователем"
def safe_float_input(prompt):
    while True:
        val = input(prompt).strip()
        try:
            return float(val)
        except ValueError:
            print("Неверный формат. Введите, пожалуйста, число.")

# Функция получения данных от пользователя
def get_user_input():
    user_data = {}
    print("\nВведите значения по признакам (без кавычек):")
    for col in X_columns_inf:
        if col in cat_features_inf:
            val = input(f"  {col} (категориально): ").strip()
            user_data[col] = val
        else:
            user_data[col] = safe_float_input(f"  {col} (число): ")
    return pd.DataFrame([user_data])

# Получаем данные от пользователя
user_input_df = get_user_input()

user_input_scaled = user_input_df.copy()
user_input_scaled[num_features_inf] = inference_scaler.transform(user_input_df[num_features_inf])

# Предсказание
pred_label = inference_model.predict(user_input_scaled)[0]
proba_all = inference_model.predict_proba(user_input_scaled)[0]

try:
    idx_yes_inf = list(inference_model.classes_).index("Да")
    proba_yes = proba_all[idx_yes_inf]
except ValueError:
    proba_yes = proba_all[1]

# Вывод результата пользователю
print("\n--- Результат предсказения ---")
if pred_label == "Да" or (not isinstance(pred_label, str) and pred_label == 1):
    print("Пожар возможен.")
else:
    print("Пожар маловероятен.")
print(f"Вероятность (класс 'Да'): {proba_yes * 100:.2f}%\n")
