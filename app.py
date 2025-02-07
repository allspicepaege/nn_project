import streamlit as st
from pages import intel_image_classification, derm_scan, training_info

st.sidebar.title('Навигация')
page = st.sidebar.selectbox('Выберите страницу', ['IntelImageClassifier', 'DermScan', 'TrainingInfo'])

if page == 'IntelImageClassifier':
    intel_image_classification.main()
elif page == 'DermScan':
    derm_scan.main()
elif page == 'TrainingInfo':
    training_info.main()