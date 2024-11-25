import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.losses import MeanSquaredError


optimizer = Adam(learning_rate=0.001)

def prepare_data(df, window_size=30):
    sales_par_jour = df.groupby(df['InvoiceDate'].dt.date)['Total_Price'].sum().reset_index()
    sales_par_jour.columns = ['ds', 'y']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_data = scaler.fit_transform(sales_par_jour['y'].values.reshape(-1, 1))
    
    joblib.dump(scaler, './Model/scaler.pkl')
    
    X, y = [], []
    for i in range(len(sales_data) - window_size):
        X.append(sales_data[i:i+window_size])
        y.append(sales_data[i+window_size])
    
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    
    return X, y

def train_lstm(df, window_size=30):
    X, y = prepare_data(df, window_size)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=MeanSquaredError())
    
    model.fit(X, y, epochs=50, batch_size=32)
    
    model.save('./Model/sales_forecast_lstm.h5')
    print("Model trained and saved as 'sales_forecast_lstm.h5'")

def train_model():
    try:
        sheets = pd.read_excel('./Data/online_retail_II.xlsx', sheet_name=None)
        df = pd.concat([sheets[sheet] for sheet in sheets], ignore_index=True)
        
        df = df[df['Customer ID'].notnull()]
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        df['Total_Price'] = df['Quantity'] * df['Price']
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        train_lstm(df)
    except Exception as e:
        print(f"Error occurred during model training: {e}")

train_model()
