import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time

# ThingSpeak API thông tin
channel_id = '2812482'  # Channel ID của bạn
read_api_key = 'SRL01GL6ZNMLKLVI'  # Thay thế bằng Read API Key của bạn
write_api_key = '5F8N9Q7AI0PQZQBH'  # Thay thế bằng Write API Key của bạn

# Đường dẫn tới file CSV
data_file_path = 'synthetic_data.csv'


# Đọc dữ liệu từ file CSV
def load_data_from_csv(file_path):
    try:
        # Đọc dữ liệu từ file CSV
        data = pd.read_csv(file_path)
        print("Dữ liệu đã được tải thành công từ file CSV!")
        return data
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu từ file: {e}")
        return None


# Huấn luyện mô hình Linear Regression
def train_model(data):
    X = data[['Vibration', 'Height', 'Soil Moisture']].values
    y = data['Warning'].values

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tạo mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Kiểm tra mô hình
    score = model.score(X_test, y_test)
    print(f'Model R^2 Score: {score}')

    return model


import requests

# ThingSpeak API thông tin
channel_id = '2812482'  # Channel ID của bạn
read_api_key = 'SRL01GL6ZNMLKLVI'  # Thay thế bằng Read API Key của bạn


# Lấy dữ liệu cuối cùng từ các field của ThingSpeak
def get_last_field_data():
    url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={read_api_key}&results=2'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Lỗi khi truy vấn dữ liệu từ ThingSpeak: {response.status_code}")
        return None, None, None  # Trả về None nếu có lỗi

    data = response.json()

    # Kiểm tra xem dữ liệu có hợp lệ không
    if 'feeds' not in data or len(data['feeds']) == 0:
        print("Không có dữ liệu hợp lệ trong ThingSpeak.")
        return None, None, None  # Trả về None nếu không có dữ liệu
    print(data)
    # Lấy dữ liệu cuối cùng (là phần tử cuối cùng trong 'feeds')
    last_feed = data['feeds'][-1]

    # Trích xuất các giá trị từ các field
    vibration = last_feed.get('field1', None)
    height = 1.5  # Giá trị cố định cho chiều cao
    soil_moisture = last_feed.get('field3', None)

    # Kiểm tra xem các giá trị có hợp lệ không
    if vibration is None or soil_moisture is None:
        print("Dữ liệu không hợp lệ từ ThingSpeak.")
        return None, None, None

    return float(vibration), float(height), float(soil_moisture)


# Cập nhật Field 4 và giữ các giá trị cũ
def update_field_4(vibration, height, soil_moisture, warning):
    url = f'https://api.thingspeak.com/update?api_key={write_api_key}&field1={vibration}&field2={height}&field3={soil_moisture}&field4={warning}'
    response = requests.get(url)

    if response.status_code == 200:
        print(f"Cập nhật thành công vào ThingSpeak! ")
    else:
        print(f"Lỗi khi cập nhật Field 4: {response.status_code}")


# Dự đoán cảnh báo từ mô hình
def predict_warning(model, vibration, height, soil_moisture):
    features = np.array([[vibration, height, soil_moisture]])

    # Kiểm tra NaN trước khi dự đoán
    if np.any(np.isnan(features)):
        print("Dữ liệu chứa NaN, bỏ qua lần này.")
        return None

    prediction = model.predict(features)

    return 1 if prediction >= 0.5 else 0


# Main function
def main():
    # Tải dữ liệu từ file CSV
    data = load_data_from_csv(data_file_path)
    if data is None:
        return

    # Huấn luyện mô hình với dữ liệu từ file CSV
    model = train_model(data)

    while True:
        # Lấy dữ liệu từ ThingSpeak
        vibration, height, soil_moisture = get_last_field_data()

        # Kiểm tra xem dữ liệu có hợp lệ không
        if vibration is None or height is None or soil_moisture is None:
            print("Dữ liệu không hợp lệ, bỏ qua lần này.")
            time.sleep(10)
            continue  # Bỏ qua vòng lặp này và tiếp tục lấy dữ liệu lần sau

        # Dự đoán cảnh báo
        warning = predict_warning(model, vibration, height, soil_moisture)

        # Kiểm tra nếu warning là None (do NaN)
        if warning is None:
            time.sleep(10)
            continue  # Bỏ qua lần này và tiếp tục lấy dữ liệu lần sau

        # Cập nhật Field 4 với giá trị cảnh báo
        update_field_4(vibration, height, soil_moisture, warning)

        # Chờ 10 giây trước khi lấy dữ liệu tiếp theo
        time.sleep(10)


if __name__ == "__main__":
    main()
