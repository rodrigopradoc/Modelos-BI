import streamlit as st
import pandas as pd
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def app():
    st.title('Predicción del precio de las acciones con el modelo Decision Tree Regressor')
    st.subheader('Preparación de la data')

    #Obteniendo la data
    ticker='MSFT'
    period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
    period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    interval = '1d' # 1d, 1m
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    df['symbol']='MSFT'
    st.write(df)

    st.subheader('Mostrando valores Fecha y precio de Cierre (Close)')
    df.index=df['Date']
    df=df.drop(['Date'],axis='columns')
    df=df[['Close']]
    st.write(df)

    #Creando la variable para predecir 'x' en el futuro
    future_days=100
    #Create a new column (target) shifted 'x' units/dayys up
    #Creamos una nueva columna 
    df['Prediction']=df[['Close']].shift(-future_days)
    df.tail(4)

    #Creando variables
    X = np.array(df.drop(['Prediction'],1))[:-future_days]
    y = np.array(df['Prediction'])[:-future_days]

    #Entrenamiento
    x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    #Aplicación de DecisionTreeRegressor
    tree = DecisionTreeRegressor().fit(x_train,y_train)
    x_future = df.drop(['Prediction'],1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    x_future

    st.subheader('Predicción de valores por DecisionTreeRegressor')
    tree_prediction = tree.predict(x_future)
    st.write(tree_prediction)

    st.subheader('Calculando el Root Men Square (RMS)')
    rms=np.sqrt(np.mean(np.power((x_future-tree_prediction),2)))
    st.write(rms)

    st.subheader('Visualización de la data')
    predictions = tree_prediction

    valid = df[X.shape[0]:]
    valid['Predictions']=predictions
    fig = plt.figure(figsize=(16,8))
    plt.title('Model Decision Tree')
    plt.xlabel('Days')
    plt.ylabel('Close Price USD($)')
    plt.plot(df['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Original Data','Valid Data', 'Predicted Data'])
    st.pyplot(fig)
