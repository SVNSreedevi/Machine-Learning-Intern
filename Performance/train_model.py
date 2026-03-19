import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_data(path='dataset/students.csv'):
    return pd.read_csv(path)


def train_and_save(path='dataset/students.csv', save_model='model.pkl', save_scaler='scaler.pkl'):
    df = load_data(path)
    X = df.drop(columns=['final_score'])
    y = df['final_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    print('Linear Regression MAE:', lr_mae, 'R2:', lr_r2)
    print('Random Forest MAE:', rf_mae, 'R2:', rf_r2)

    # Best model selection
    best = rf if rf_mae <= lr_mae else lr
    joblib.dump(best, save_model)
    joblib.dump(scaler, save_scaler)
    print('Saved model to', save_model, 'and scaler to', save_scaler)


if __name__ == '__main__':
    train_and_save()
