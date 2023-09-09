import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Загрузка данных
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

data = load_data('dataset.json')

# Построение модели
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=((7,7),(None, 3, None, None))),
    layers.Dense(32, activation='relu'),
    layers.Dense(output_shape=(), activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
def build_routes(stations, full_timetable):
    return None

# Преобразование времени в числовой формат
def convert_time_to_numbers(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def convert_data_select_to_train_data(data_select):
    stations = data_select['stations']
    stations = np.array([stations[station_name] for station_name in data_select['stations']])
    
    full_timetable = data_select['full_timetable']
    tensor_data = {}

    for key, value in full_timetable.items():
        route = np.array(value["route"], dtype=int)
        free_carriage = np.array(value["free_carriage"], dtype=int)
        timetable = [convert_time_to_numbers(time_range.split(' - ')[0]) for time_range in value["timetable"]]
        timetable = np.array(timetable, dtype=int)
        tensor_data[key] = {"route": route, "free_carriage": free_carriage, "timetable": timetable}

    train_data = [stations, tensor_data]

    return train_data

def model_fit(model, data_select):
    train_data = convert_data_select_to_train_data(data_select)
    
    validation_data = build_routes(train_data[0], train_data[1])

    # Тренировка модели
    model.fit(train_data, epochs=10, validation_data=validation_data)

for data_select in data:
    model_fit(model, data_select)

