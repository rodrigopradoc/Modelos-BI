import streamlit as st
from apps import ModeloSVR1,DecisionTreeRegressor # import your app modules here

st.title("Inteligencia de Negocios - Equipo C")
st.subheader("Modelos SVR y DecisionTreeRegressor")
st.write("Se muestra la aplicación de los modelos para la predicción en la bolsa de valores")
st.write("---")

ModeloSVR1.app()
st.write("---")
DecisionTreeRegressor.app()


