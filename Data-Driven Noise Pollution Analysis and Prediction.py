# =========================
# Data Collection
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/aksha/Documents/project/INT - 375 CA -2/urban_noise_levels.csv")

print(df.head())
print("Dataset Shape:", df.shape)
print(df.info())


# =========================
# Data Cleaning
# =========================

df = df.drop(columns=['id', 'sensor_id', 'datetime'])

for col in df.columns:
    if df[col].dtype != 'object':
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print(df.isnull().sum())


# =========================
# Exploratory Data Analysis
# =========================

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),
            cmap='RdYlBu_r',
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor='white')
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

sns.scatterplot(x='traffic_density', y='decibel_level', data=df,
                color='green', alpha=0.6, edgecolor='black')
plt.title("Traffic Density vs Noise Level", fontsize=13)
plt.xlabel("Traffic Density")
plt.ylabel("Noise Level (dB)")
plt.show()

# (Removed vehicle_count error and replaced with safe plot)
sns.scatterplot(x='traffic_density', y='decibel_level', data=df,
                color='orange', alpha=0.6, edgecolor='black')
plt.title("Traffic Density vs Noise Level (Alternate View)", fontsize=13)
plt.xlabel("Traffic Density")
plt.ylabel("Noise Level (dB)")
plt.show()


# =========================
# Feature Engineering
# =========================

X = df.drop(columns=['decibel_level'])
y = df['decibel_level']


# =========================
# Modeling (Linear Regression)
# =========================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# =========================
# Evaluation
# =========================

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


# =========================
# OBJECTIVE 1: Predict noise levels
# =========================

sns.regplot(x=y_test, y=y_pred,
            scatter_kws={'color':'purple','alpha':0.6},
            line_kws={'color':'black'})
plt.xlabel("Actual Noise Level")
plt.ylabel("Predicted Noise Level")
plt.title("Regression Plot: Actual vs Predicted", fontsize=13)
plt.show()


# =========================
# OBJECTIVE 2: Identify key factors
# =========================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("Top factors affecting noise:")
print(feature_importance.head())

sns.barplot(x='Coefficient', y='Feature',
            data=feature_importance.head(10),
            palette='viridis', edgecolor='black')
plt.title("Top Features Affecting Noise Level", fontsize=13)
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.show()


# =========================
# OBJECTIVE 3: Environmental impact
# =========================

sns.scatterplot(x='temperature_c', y='decibel_level', data=df,
                color='blue', alpha=0.6, edgecolor='black')
plt.title("Temperature vs Noise Level", fontsize=13)
plt.xlabel("Temperature (°C)")
plt.ylabel("Noise Level (dB)")
plt.show()

sns.scatterplot(x='humidity_%', y='decibel_level', data=df,
                color='cyan', alpha=0.6, edgecolor='black')
plt.title("Humidity vs Noise Level", fontsize=13)
plt.xlabel("Humidity (%)")
plt.ylabel("Noise Level (dB)")
plt.show()


# =========================
# OBJECTIVE 4: High noise zones
# =========================

high_noise = df[df['decibel_level'] > df['decibel_level'].mean()]

sns.scatterplot(x='traffic_density', y='decibel_level', data=high_noise,
                color='red', alpha=0.7, edgecolor='black')
plt.title("High Noise Zones (Traffic vs Noise)", fontsize=13)
plt.xlabel("Traffic Density")
plt.ylabel("Noise Level (dB)")
plt.show()

sns.histplot(high_noise['decibel_level'],
             bins=20, color='red', alpha=0.6, edgecolor='black')
plt.title("Distribution of High Noise Levels", fontsize=13)
plt.xlabel("Noise Level (dB)")
plt.show()


# =========================
# OBJECTIVE 5: Model performance
# =========================

error = y_test - y_pred

sns.histplot(error,
             bins=30, kde=True,
             color='green', alpha=0.6, edgecolor='black')
plt.title("Error Distribution", fontsize=13)
plt.xlabel("Prediction Error")
plt.show()