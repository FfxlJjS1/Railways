import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Загрузка данных
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

data = load_data('1.json')

max_trains = 0

for data_select in data:
    max_trains = max(max_trains, len(data_select['full_timetable']))

# Построение модели
input_1 = layers.Input(shape=(7, 7))  # Вход для станций
input_2 = layers.Input(shape=(max_trains, 3, 7))  # Вход для расписания

flatten_1 = layers.Flatten()(input_1)
flatten_2 = layers.Flatten()(input_2)

concatenated = layers.Concatenate()([flatten_1, flatten_2])

hidden_1 = layers.Dense(512, activation='relu')(concatenated)
hidden_2 = layers.Dense(256, activation='relu')(hidden_1)
hidden_3 = layers.Dense(256, activation='relu')(hidden_2)

# Создайте выходной слой с учетом максимального числа поездов и станций
output = layers.Dense(7*max_trains, activation='softmax')(hidden_3) # lines are trains and columns are stations where strains will trasport carriage  by own timetable

model = models.Model(inputs=[input_1, input_2], outputs=output)

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
def build_routes(stations, full_timetable):
    res_matr = np.array([[0]*7 for i in range(len(full_timetable))])

    for train_index in range(len(full_timetable)):
        timetable = full_timetable[train_index]

        for routeIndex in range(len(timetable[0])-1):
            free_carriage = timetable[1][routeIndex]

            current_station_number = timetable[0][routeIndex]
            next_station_number = timetable[0][routeIndex+1]

            for_next_station_carriage_count = stations[current_station_number-1][next_station_number-1]

            transport_carriage_count = min(free_carriage, for_next_station_carriage_count)

            stations[current_station_number-1][next_station_number-1] = stations[current_station_number-1][next_station_number-1] - transport_carriage_count

            res_matr[train_index][next_station_number-1] = transport_carriage_count

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
    stations = np.array([[int(station) for station in stations[station_name]] for station_name in data_select['stations']])
    
    full_timetable = data_select['full_timetable']
    tensor_data = []
    train_numbers = []

    for key, value in full_timetable.items():
        train_numbers += [key]
        route = np.array([int(i) for i in value["route"]], dtype=int)
        free_carriage = np.array([int(i) for i in value["free_carriage"]], dtype=int)
        timetable = [convert_time_to_numbers(time_range.split(' - ')[0]) for time_range in value["timetable"]]
        timetable = np.array([int(i) for i in timetable], dtype=int)
        tensor_data += [[
            np.pad(route, ((0, 7 - len(route))), mode='constant', constant_values=0), 
            np.pad(free_carriage, ((0, 7 - len(free_carriage))), mode='constant', constant_values=0), 
            np.pad(timetable, ((0, 7 - len(timetable))), mode='constant', constant_values=0)]]

    train_data = [np.array(stations), np.array(tensor_data)]

    return [train_data, train_numbers]

def model_fit(model, data_select):
    train_data_1 = convert_data_select_to_train_data(data_select)
    train_data, train_numbers = train_data_1[0], train_data_1[1]
    
    validation_data = build_routes(train_data[0], train_data[1])

    # Тренировка модели
    model.fit(x=train_data,y=validation_data, epochs=10)

for data_select in data:
    model_fit(model, data_select)

 