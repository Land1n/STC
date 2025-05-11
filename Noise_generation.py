from math import floor
import numpy as np
from convolution import *

def add_awgn_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Добавляет аддитивный белый гауссовский шум (AWGN) к сигналу.

    Параметры:
    ----------
    signal : np.ndarray
        Входной сигнал (1D массив).
    snr_db : float
        Желаемое соотношение сигнал/шум в децибелах (dB).

    Возвращает:
    -----------
    noisy_signal : np.ndarray
        Сигнал с добавленным шумом (той же длины, что и входной).
    """

    # Вычисляем мощность сигнала
    signal_power = np.mean(np.abs(signal) ** 2)

    # Переводим SNR из dB в линейный масштаб
    snr_linear = 10 ** (snr_db / 10)

    # Вычисляем требуемую мощность шума
    noise_power = signal_power / snr_linear

    # Генерируем шум с нормальным распределением (std = sqrt(noise_power))
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)

    # Добавляем шум к сигналу
    noisy_signal = signal + noise

    return noisy_signal
# Пример
"""
t, pulse = rectangular_pulse(start=-0.5,
                             duration=1,
                             amplitude=1,
                             t_start=-1,
                             t_end=1,
                             fs=1000)

t1, pulse1 = rectangular_pulse(start=0,
                             duration=1,
                             amplitude=1,
                             t_start=-1,
                             t_end=1,
                             fs=1000)
#t, pulse1 = triangular_pulse(peak_time=0.5,t_start =0,t_end=2, A=1,fs=1000)
#t1, pulse1 =  gaussian_pulse(μ=1, σ=0.3, A=1, t_start=0, t_end=2, fs=2000)

#a=convolution(pulse,pulse1)
pulse = add_awgn_noise(pulse, -17)
tX,b=correlation(pulse1,t1,pulse,t)
delay = find_max_with_time(b, tX)
print(len(tX))
print(len(b))
pm = PlotManager()

data_dict = {
    'x':t,
        'y1':  pulse,
        #'y2': pulse1
    }
df = pm.load_data(data=data_dict)
# Построение графиков с автоматическими точками по оси X
pm.plot(
        data=df, #None, для скопированных данных
        x_column='x',  # None, для автоматического расположения точек, 0 для скопированных данных
        y_column='y1',
        labels={'y1': 'Второй сигнал', 'y2': 'Второй сигнал'},
        title='Сигналы используемые в корреляции',
        xlabel='Время, с',
        ylabel='Значение',
        marker=None,
        color="Blue",
        linewidth=1.5,
        xstart=-1,
        ystart=None,
        xticks=0.25,
        yticks=None,

        ask_input=False #True, если надо вводить скопированные столбцы
    )
data_dict = {
    'x':tX,
        'y1':  b,
        #'y2': pulse1
    }
df = pm.load_data(data=data_dict)
# Построение графиков с автоматическими точками по оси X
pm.plot(
        data=df, #None, для скопированных данных
        x_column='x',  # None, для автоматического расположения точек, 0 для скопированных данных
        y_column='y1',
        labels={'y1': f'Сдвиг: {floor(delay*1000)/1000} c', 'y2': 'Второй сигнал'},
        title='Корреляция',
        xlabel='Время, с',
        ylabel='Значение',
        marker=None,
        color="Green",
        linewidth=1.5,
        xstart=-2,
        ystart=None,
        xticks=0.5,
        yticks=None,

        ask_input=False #True, если надо вводить скопированные столбцы
    )
"""