import streamlit as st
from pages.intel_image_classification import main as intel_image_classification_main
from pages.derm_scan import main as derm_scan_main
from pages.training_info import main as training_info_main

# Навигация по страницам
st.sidebar.title('Навигация')
page = st.sidebar.selectbox('Выберите страницу', ['Сканирование дермы', 'Классификация изображений Intel', 'Информация об обучении'])

if page == 'Сканирование дермы':
    derm_scan_main()
elif page == 'Классификация изображений Intel':
    intel_image_classification_main()
elif page == 'Информация об обучении':
    training_info_main()
