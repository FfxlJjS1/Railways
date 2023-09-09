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
    layers.Dense(output_shape=(None,7), activation='softmax') # lines are trains and columns are stations where strains will trasport carriage  by own timetable
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
def build_routes(stations, full_timetable):
    res_matr = [[0]*7 for i in range(len(full_timetable))]

    for train_number in full_timetable:
        timetable = full_timetable[train_number]

        for routeIndex in range(len(timetable['route'])-1):
            free_carriage = timetable['free_carriage'][routeIndex]

            current_station_number = int(timetable['route'][routeIndex])
            next_station_number = int(timetable['route'][routeIndex+1])

            for_next_station_carriage_count = int(stations[current_station_number-1][next_station_number-1])

            transport_carriage_count = min(free_carriage, for_next_station_carriage_count)

            stations[current_station_number-1][next_station_number-1] = str(int(stations[current_station_number-1][next_station_number-1]) - transport_carriage_count)

            res_matr[list(full_timetable.keys()).index(train_number)][next_station_number-1] = transport_carriage_count

    # 1. Самый нечастый (короткий, по узлу) / самое большое количество за проезд. Приоритет из-за количества за раз на близкий, иначе свалка

    # Переброска/прямой путь по самому большому кол-во вагонов в результате. По короткому пути из-за узких узлов поездов


    # Прямые пути / учет результирующей переброски... Пересечение возможного количества вагонов для переброски на каждом пути дальнего пути
    # Приоритет по количеству на близкий и по узлу

    # Пересечение / те, которые не перебросить по прямой. Пересечение возможного количества вагонов для переброски на каждом пути переброски


    return res_matr

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

 