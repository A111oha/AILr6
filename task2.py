import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Завантаження даних
url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
df = pd.read_csv(url)

# Попередній перегляд даних
print(df.head())

# Видалення лише тих колонок, які існують у даних
columns_to_drop = ['insert_date']  # Видаляємо тільки існуючу колонку
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

# Видалення рядків із пропущеними значеннями
df = df.dropna()

# Кодування текстових даних у числовий формат
encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = encoder.fit_transform(df[column])

# Визначення ознак (X) та мітки класу (y)
X = df.drop('price', axis=1)  # Усі колонки, окрім ціни
# Перетворення ціни на категорії
bins = [0, 50, 100, 150, 200, float('inf')]  # Діапазони цін
labels = [0, 1, 2, 3, 4]  # Категорії
df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)

# Використання категорій як мітки класів
y = df['price_category']

# Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Використання наївного байєсівського класифікатора (Гауссівський)
model = GaussianNB()
model.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
y_pred = model.predict(X_test)

# Оцінка моделі
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.2f}")
print("Класифікаційний звіт:")
print(classification_report(y_test, y_pred))

