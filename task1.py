import pandas as pd

#  датафрейм з даних таблиці
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}
df = pd.DataFrame(data)

#  умови для прогнозу
conditions = {"Outlook": "Overcast", "Humidity": "High", "Wind": "Weak"}

# Підрахуємо загальну ймовірність кожного класу ("Yes" або "No")
play_yes = len(df[df["Play"] == "Yes"]) / len(df)
play_no = len(df[df["Play"] == "No"]) / len(df)

# Функція для розрахунку умовних ймовірностей
def calculate_probabilities(df, conditions, target_class):
    subset = df[df["Play"] == target_class]
    probabilities = 1
    for feature, value in conditions.items():
        probabilities *= len(subset[subset[feature] == value]) / len(subset)
    return probabilities

# Умовні ймовірності для "Yes" та "No"
prob_yes = calculate_probabilities(df, conditions, "Yes") * play_yes
prob_no = calculate_probabilities(df, conditions, "No") * play_no

# Нормалізуємо ймовірності
total_prob = prob_yes + prob_no
prob_yes_normalized = prob_yes / total_prob
prob_no_normalized = prob_no / total_prob

# Відображення відсотків
print(f"P(Yes) = {prob_yes_normalized:.2f} ({prob_yes_normalized * 100:.2f}%)")
print(f"P(No) = {prob_no_normalized:.2f} ({prob_no_normalized * 100:.2f}%)")
