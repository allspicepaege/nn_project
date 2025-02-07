import streamlit as st
from PIL import Image

def main():
    # Отображение логотипа
    logo = Image.open('images/logo_training.jpg')
    st.image(logo, width=800)

    # Интерфейс Streamlit
    st.title('TrainingInfo: Информация об обучении моделей')

    # Информация о датасете
    st.subheader('Информация о датасете')
    st.write('''
    ## Intel Image Classification Dataset
    - **Количество классов**: 6 (здания, лес, ледник, гора, море, улица)
    ''')

    # Графики обучения
    st.subheader('График обучения - accuracy')
    training_graph = Image.open('images/metrics_accuracy.jpg')
    st.image(training_graph, caption='accuracy', use_container_width=True)

    st.subheader('График обучения - f1-score')
    training_graph = Image.open('images/metrics_f1-score.jpg')
    st.image(training_graph, caption='f1-score', use_container_width=True)

    st.subheader('График обучения - Confusion Matrix')
    training_graph = Image.open('images/metrics_Confusion Matrix.jpg')
    st.image(training_graph, caption='Confusion Matrix', use_container_width=True)

    # Информация о датасете
    st.subheader('Информация о датасете')
    st.write('''
    ## Derm_Scan Dataset
    - **Количество классов**: 2 (доброкачественные и злокачественные)
    ''')

    # Графики обучения
    st.subheader('График обучения - accuracy')
    training_graph = Image.open('images/metrics_accuracy2.jpg')
    st.image(training_graph, caption='accuracy', use_container_width=True)

    st.subheader('График обучения - f1-score')
    training_graph = Image.open('images/metrics_f1-score2.jpg')
    st.image(training_graph, caption='f1-score', use_container_width=True)

    st.subheader('График обучения - Confusion Matrix')
    training_graph = Image.open('images/metrics_Confusion Matrix2.jpg')
    st.image(training_graph, caption='Confusion Matrix', use_container_width=True)

if __name__ == "__main__":
    main()
