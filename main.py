import streamlit as st
import utils.funciones as fc

# configurar página
fc.config_page()

st.title('Predicción de ataques de apoplejía')

# menu
menu = st.sidebar.selectbox('Datos COVID-19', ['Portada', 'Datos', 'Modelos', 'Predecir'])

if menu == 'Portada':
    fc.portada()

elif menu == 'Datos':
    fc.datos()

elif menu == 'Modelos':
    fc.modelos()

else:
    fc.predecir()