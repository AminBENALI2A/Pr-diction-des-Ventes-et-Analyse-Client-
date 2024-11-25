from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from flask import Flask, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

import datetime as dt
import json
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/OriginalData', methods=['GET'])
def return_all_sheets():
    try:
        sheets = pd.read_excel('./Data/online_retail_II.xlsx')
        sheets_data = {}

        for sheet_name, sheet_df in sheets.items():
            sheet_df = sheet_df[sheet_df['Customer ID'].notnull()]
            sheet_df = sheet_df[(sheet_df['Quantity'] > 0) & (sheet_df['Price'] > 0)]

            sheet_df['Total_Price'] = sheet_df['Quantity'] * sheet_df['Price']
            sheet_df['InvoiceDate'] = pd.to_datetime(sheet_df['InvoiceDate'])

            sheets_data[sheet_name] = sheet_df.to_dict(orient="records")

        return jsonify(sheets_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def calculate_rfm(df):
    reference_date = dt.datetime(2011, 12, 10)
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'Invoice': 'nunique',
        'Total_Price': 'sum'
    }).reset_index()
    
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    return rfm


def cluster_rfm(rfm, n_clusters=5):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm


@app.route('/RFM_Clustering', methods=['GET'])
def rfm_clustering():
    try:
        sheets = pd.read_excel('./Data/online_retail_II.xlsx', sheet_name=None)
        df = pd.concat([sheets[sheet] for sheet in sheets], ignore_index=True)
        
        df = df[df['Customer ID'].notnull()]
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        df['Total_Price'] = df['Quantity'] * df['Price']
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        rfm = calculate_rfm(df)
        
        n_clusters = int(request.args.get('n_clusters', 5))
        clustered_rfm = cluster_rfm(rfm, n_clusters)
        
        return jsonify(clustered_rfm.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ForecastNextQuarter', methods=['GET'])
def forecast_next_quarter():
    try:
        model = load_model('./Model/sales_forecast_lstm.h5', custom_objects={"MeanSquaredError": MeanSquaredError()})
        scaler = joblib.load('./Model/scaler.pkl')

        sheets = pd.read_excel('./Data/online_retail_II.xlsx', sheet_name=None)
        df = pd.concat([sheets[sheet] for sheet in sheets], ignore_index=True)

        df = df[df['Customer ID'].notnull()]
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        df['Total_Price'] = df['Quantity'] * df['Price']
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        df['Date'] = df['InvoiceDate'].dt.date
        sales_par_jour = df.groupby('Date')['Total_Price'].sum().reset_index()

        sales_par_jour.columns = ['Date', 'Total_Sales']
        sales_data = sales_par_jour['Total_Sales'].values

        scaled_sales = scaler.transform(sales_data.reshape(-1, 1))

        if len(scaled_sales) < 30:
            return jsonify({"error": "Not enough data for prediction. At least 30 days of data are required."}), 400
        
        X_input = scaled_sales[-30:].reshape(1, 30, 1)

        start_date = sales_par_jour['Date'].iloc[-1]
        start_date = pd.to_datetime(start_date)

        predictions = []
        for i in range(90):
            pred = model.predict(X_input, verbose=0)
            pred_original = scaler.inverse_transform(pred)[0][0]
            prediction_date = start_date + timedelta(days=i + 1)
            predictions.append({"date": prediction_date.strftime("%Y-%m-%d"), "predicted_sales": round(float(pred_original),3)})
            
            X_input = np.append(X_input[:, 1:, :], [pred], axis=1)
            X_input = X_input[:, :30, :]

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
