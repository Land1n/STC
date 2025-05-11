import numpy as np
from scipy import interpolate
from style_f import *
from generationSignal import *

#Свертка
import numpy as np
from scipy import interpolate

def convolution(firstSignal, timeline1, secondSignal, timeline2):
    """
    Вычисляет свертку двух сигналов с возможным временным сдвигом.

    Параметры:
    ----------
    firstSignal : np.ndarray
        Первый сигнал.
    timeline1 : np.ndarray
        Временная ось первого сигнала (должна быть монотонной).
    secondSignal : np.ndarray
        Второй сигнал.
    timeline2 : np.ndarray
        Временная ось второго сигнала (должна быть монотонной).

    Возвращает:
    -----------
    time_shifts : np.ndarray
        Ось временных сдвигов (в секундах).
    convolution_func : np.ndarray
        Нормированная свертка (максимум = 1).
    """

    # Проверка, что временные оси монотонны
    if not (np.all(np.diff(timeline1) > 0) or not (np.all(np.diff(timeline2) > 0))):
        raise ValueError("Временные оси должны быть строго возрастающими!")

    # Объединяем временные оси, чтобы найти общий диапазон
    min_t = max(timeline1[0], timeline2[0])  # Начало перекрытия
    max_t = min(timeline1[-1], timeline2[-1])  # Конец перекрытия

    if min_t >= max_t:
        raise ValueError("Сигналы не перекрываются по времени!")

    # Новая временная ось (в пределах перекрытия)
    common_time = np.linspace(min_t, max_t, min(len(timeline1), len(timeline2)))

    # Интерполяция сигналов на общую ось
    interp1 = interpolate.interp1d(timeline1, firstSignal, kind='linear', fill_value=0.0, bounds_error=False)
    interp2 = interpolate.interp1d(timeline2, secondSignal, kind='linear', fill_value=0.0, bounds_error=False)

    aligned_signal1 = interp1(common_time)
    aligned_signal2 = interp2(common_time)

    # Вычисление свертки (режим 'full' дает все возможные сдвиги)
    conv = np.convolve(aligned_signal1, aligned_signal2, mode='full')

    # Нормировка на максимум
    conv = conv / np.max(conv) if np.max(conv) != 0 else conv

    # Ось временных сдвигов (в секундах)
    dt = common_time[1] - common_time[0]  # Шаг времени
    shifts = np.arange(-len(aligned_signal2) + 1, len(aligned_signal1)) * dt

    return shifts, conv

#Корреляция
def correlation(firstSignal, timeline1, secondSignal, timeline2):
    """
    Вычисляет кросс-корреляцию двух сигналов с возможным временным сдвигом.

    Параметры:
    ----------
    firstSignal : np.ndarray
        Первый сигнал.
    timeline1 : np.ndarray
        Временная ось первого сигнала (должна быть монотонной).
    secondSignal : np.ndarray
        Второй сигнал.
    timeline2 : np.ndarray
        Временная ось второго сигнала (должна быть монотонной).

    Возвращает:
    -----------
    time_lags : np.ndarray
        Ось временных задержек (в секундах).
    correlation_func : np.ndarray
        Нормированная кросс-корреляция (максимум = 1).
    """

    # Проверка, что временные оси монотонны
    if not (np.all(np.diff(timeline1) > 0) or not (np.all(np.diff(timeline2) > 0))):
        raise ValueError("Временные оси должны быть строго возрастающими!")

    # Объединяем временные оси, чтобы найти общий диапазон
    min_t = max(timeline1[0], timeline2[0])  # Начало перекрытия
    max_t = min(timeline1[-1], timeline2[-1])  # Конец перекрытия

    if min_t >= max_t:
        raise ValueError("Сигналы не перекрываются по времени!")

    # Новая временная ось (в пределах перекрытия)
    common_time = np.linspace(min_t, max_t, min(len(timeline1), len(timeline2)))

    # Интерполяция сигналов на общую ось
    interp1 = interpolate.interp1d(timeline1, firstSignal, kind='linear', fill_value=0.0, bounds_error=False)
    interp2 = interpolate.interp1d(timeline2, secondSignal, kind='linear', fill_value=0.0, bounds_error=False)

    aligned_signal1 = interp1(common_time)
    aligned_signal2 = interp2(common_time)

    # Вычисление кросс-корреляции (режим 'full' дает все возможные сдвиги)
    corr = np.correlate(aligned_signal1, aligned_signal2, mode='full')

    # Нормировка на максимум (можно заменить на энергию сигналов)
    corr = corr / np.max(corr)

    # Ось задержек (в секундах)
    dt = common_time[1] - common_time[0]  # Шаг времени
    lags = np.arange(-len(aligned_signal2) + 1, len(aligned_signal1)) * dt

    return lags, corr


def find_max_with_time(b, tX):
    """
    Находит максимальное значение в массиве `b`, его индекс и соответствующее время из `tX`.

    Параметры:
    ----------
    b : np.ndarray
        Массив данных (например, сигнал).
    tX : np.ndarray
        Массив времени, соответствующий `b` (должен быть той же длины).

    Возвращает:
    -----------
    max_val : float
        Максимальное значение в `b`.
    max_idx : int
        Индекс максимального значения.
    max_time : float
        Время, соответствующее максимуму.
    """
    if len(b) != len(tX):
        raise ValueError("Массивы `b` и `tX` должны быть одинаковой длины!")

    max_idx = np.argmax(b)  # Индекс максимального элемента
    max_val = b[max_idx]  # Максимальное значение
    max_time = tX[max_idx]  # Время максимума

    return  max_time
#Пример:
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
                             t_end=2,
                             fs=1000)
#t, pulse1 = triangular_pulse(peak_time=0.5,t_start =0,t_end=2, A=1,fs=1000)
#t1, pulse1 =  gaussian_pulse(μ=1, σ=0.3, A=1, t_start=0, t_end=2, fs=2000)

#a=convolution(pulse,pulse1)

tX,b=correlation(pulse1,t1,pulse,t)
print(len(tX))
print(len(b))
pm = PlotManager()

data_dict = {
    'x':tX,
        'y1': b,
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
        color="saddlebrown",
        linewidth=1.5,
        xstart=None,
        ystart=None,
        xticks=None,
        yticks=None,

        ask_input=False #True, если надо вводить скопированные столбцы
    )
"""