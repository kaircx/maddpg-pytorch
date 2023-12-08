import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg') 
def load_tensorboard_data(logdir):
    # TensorBoardのデータを読み込む
    ea = event_accumulator.EventAccumulator(logdir,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        values = [event.value for event in events]
        data[tag] = values

    return data

def average_runs(directories):
    all_data = {}

    for dir in directories:
        data = load_tensorboard_data(dir)
        for tag, values in data.items():
            if tag not in all_data:
                all_data[tag] = []
            all_data[tag].append(values)

    averaged_data = {}
    for tag, values in all_data.items():
        # 各タグの値のリストの長さを揃える
        min_length = min(len(v) for v in values)
        trimmed_values = [v[:min_length] for v in values]

        # 平均を計算
        averaged_data[tag] = np.mean(trimmed_values, axis=0)

    return averaged_data

def moving_average(data, window_size):
    """ 移動平均を計算する """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_data(averaged_data, averaged_data_non, window_size=10):
    for tag in averaged_data:
        plt.figure()
        # データの長さを揃える
        min_length = min(len(averaged_data[tag]), len(averaged_data_non.get(tag, [])))
        values = averaged_data[tag][:min_length]
        values_non = averaged_data_non.get(tag, [])[:min_length]

        # スムージングを適用
        smoothed_values = moving_average(values, window_size)
        smoothed_values_non = moving_average(values_non, window_size) if len(values_non) > 0 else []

        # データをプロット
        plt.plot(smoothed_values, label='Averaged Data (Smoothed)', linewidth=0.5)
        if len(smoothed_values_non) > 0:
            plt.plot(smoothed_values_non, label='Averaged Data Non (Smoothed)', linewidth=0.5)

        plt.title(tag)
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.savefig('b.png')
        plt.close()

base_path = 'models/direction/12.8.main9/run{}/logs/events.out.tfevents.1702018878.dlbox3090'
directories = [base_path.format(i) for i in range(1, 6)]  # run11からrun15まで

base_path_non = 'models/direction/12.8.main9/run{}/logs/events.out.tfevents.1702018905.dlbox3090'
directories_non = [base_path_non.format(i) for i in range(6, 11)]  # run16からrun20まで

averaged_data = average_runs(directories)
averaged_data_non = average_runs(directories_non)
plot_data(averaged_data, averaged_data_non)