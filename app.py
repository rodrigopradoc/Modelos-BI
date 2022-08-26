import streamlit as st
from apps import ModeloSVR1,DecisionTreeRegressor # import your app modules here
from streamlit_option_menu import option_menu

st.title("Inteligencia de Negocios - Equipo C")
st.subheader("Modelos SVR y DecisionTreeRegressor")
st.write("Se muestra la aplicación de los modelos para la predicción en la bolsa de valores")
st.write("---")

with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=["Modelo SVR","DecisionTreeRegressor"],
    )

if selected=="Modelo SVR":
    ModeloSVR1.app()
if selected=="DecisionTreeRegressor":
    DecisionTreeRegressor.app()


#ModeloSVR1.app()
st.write("---")
#DecisionTreeRegressor.app()


