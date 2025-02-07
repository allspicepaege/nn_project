import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
import requests
from io import BytesIO
import time  # Импортируем модуль time для измерения времени

def main():
    # Загрузка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # Изменение последнего слоя для 6 классов
    model.load_state_dict(torch.load('models/resnet_weights_IIC.pt', map_location=device))
    model.eval()
    model.to(device)

    # Преобразование изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Отображение логотипа (увеличен размер)
    logo = Image.open('images/logo_intel.jpg')
    st.image(logo, width=800)  # Увеличен размер логотипа

    # Интерфейс Streamlit
    st.title('IntelImageClassifier: Классификация изображений Intel Image Classification')

    # Описание перед загрузкой изображения
    st.write("Загрузите изображение, чтобы увидеть, к какому классу оно относится.")
    st.write("Классы:")
    st.write("0 - здания")
    st.write("1 - лес")
    st.write("2 - ледник")
    st.write("3 - гора")
    st.write("4 - море")
    st.write("5 - улица")

    # Загрузка изображения из файла
    uploaded_files = st.file_uploader("Выберите изображение (или несколько)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Загрузка изображения по ссылке
    image_urls = st.text_area("Введите URL изображений (каждый на новой строке)")
    image_urls = image_urls.splitlines()

    # Обработка загруженных изображений
    all_images = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            all_images.append(image)

    if image_urls:
        for image_url in image_urls:
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()  # Проверка на успешный ответ
                image = Image.open(BytesIO(response.content))
                all_images.append(image)
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка загрузки изображения по ссылке: {e}")

    if all_images:
        for image in all_images:
            st.image(image, caption='Загруженное изображение', use_container_width=True)  # Используем use_container_width
            image = transform(image).unsqueeze(0)

            start_time = time.time()  # Засекаем время перед классификацией

            with torch.no_grad():
                image = image.to(device)
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            end_time = time.time()  # Засекаем время после классификации
            elapsed_time = end_time - start_time  # Вычисляем время, затраченное на классификацию

            st.write(f'Предсказанный класс: {predicted.item()}')
            st.write(f'Время классификации: {elapsed_time:.4f} сек.')  # Выводим время с точностью до 4 знаков после запятой

if __name__ == "__main__":
    main()