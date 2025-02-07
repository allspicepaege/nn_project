import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
import requests
from io import BytesIO

def main():
    # Загрузка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Изменение последнего слоя для бинарной классификации
    model.load_state_dict(torch.load('models/resnet_weights_SCMB.pt', map_location=device))
    model.eval()
    model.to(device)

    # Преобразование изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Отображение логотипа
    logo = Image.open('images/logo_skin.jpg')
    st.image(logo, width=200)

    # Интерфейс Streamlit
    st.title('DermScan: Классификация образований на коже')
    st.write("Загрузите изображение или введите URL изображения для классификации.")

    # Загрузка изображений
    uploaded_files = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    image_url = st.text_input("Введите URL изображения")

    # Описание классов
    class_names = ['Доброкачественные', 'Злокачественные']

    if uploaded_files or image_url:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', use_column_width=True)
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                image = image.to(device)
                outputs = model(image)
                predicted = torch.sigmoid(outputs).item()  # Применение сигмоиды для бинарной классификации
                st.write(f'Предсказанный класс: {"Злокачественные" if predicted > 0.5 else "Доброкачественные"}')
                st.write(f'Вероятность: {predicted:.4f}')

        if image_url:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Загруженное изображение', use_column_width=True)
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                image = image.to(device)
                outputs = model(image)
                predicted = torch.sigmoid(outputs).item()  # Применение сигмоиды для бинарной классификации
                st.write(f'Предсказанный класс: {"Злокачественные" if predicted > 0.5 else "Доброкачественные"}')
                st.write(f'Вероятность: {predicted:.4f}')

if __name__ == "__main__":
    main()
