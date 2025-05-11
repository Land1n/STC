import numpy as np
    

def gaussian_pulse(μ=0, σ=1, A=1, t_start=None, t_end=None, fs=1000):
    """
    Параметры:
        μ (float): Среднее (центр импульса).
        σ (float): Стандартное отклонение (ширина импульса).
        A (float): Амплитуда импульса.
        t_start (float): Начало временного интервала (по умолчанию μ - 5σ).
        t_end (float): Конец временного интервала (по умолчанию μ + 5σ).
        fs (int): Количество точек для построения.
    """
    if σ <= 0:
        raise ValueError("σ должно быть положительным.")
    
    # Определение диапазона времени, если не задано
    if t_start is None:
        t_start = μ - 5 * σ
    if t_end is None:
        t_end = μ + 5 * σ
    
    # Создание временной оси
    t = np.linspace(t_start, t_end, fs)
    
    # Формула гауссовского импульса
    gaussian = A * np.exp(-(t - μ)**2 / (2 * σ**2))
    
    return t,gaussian

def rectangular_pulse(start=0, duration=1, amplitude=1, 
                     t_start=None, t_end=None, fs=1000):
    """
    Генерирует прямоугольный импульс
    
    Параметры:
        start (float): Начало импульса (секунды)
        duration (float): Длительность импульса (секунды)
        amplitude (float): Амплитуда импульса
        t_start (float): Начало временного интервала (по умолчанию start - 2*duration)
        t_end (float): Конец временного интервала (по умолчанию start + 3*duration)
        fs (int): Количество точек для построения
    """
    # Определение временного диапазона
    if t_start is None:
        t_start = start - 3*duration
    if t_end is None:
        t_end = start + 3*duration
        
    # Создание временной оси
    t = np.linspace(t_start, t_end, int((duration-start)*fs) )
    
    # Создание прямоугольного импульса
    pulse = np.where((t >= start) & (t <= (start + duration)), A, 0)

    return t, pulse


def triangular_pulse(peak_time=0, rise_angle=None, fall_angle=None, A=1, 
                    t_start=None, t_end=None, fs=1000):
    """
    Генерирует треугольный импульс с заданными углами наклона сторон в радианах.

    Параметры:
        peak_time (float): Время достижения максимума импульса.
        rise_angle (float): Угол наклона левого склона в радианах (0 < angle < π/2).
        fall_angle (float): Угол наклона правого склона в радианах (0 < angle < π/2).
        A (float): Амплитуда импульса.
        t_start (float): Явное задание начала сигнала (имеет приоритет над rise_angle).
        t_end (float): Явное задание конца сигнала (имеет приоритет над fall_angle).
        fs (int): Количество точек для построения.

    Возвращает:
        t (ndarray): Временная ось.
        triangular (ndarray): Значения треугольного сигнала.

    Вызывает:
        ValueError: При некорректных параметрах.
    """
    # Проверка и вычисление временных границ
    if t_start is None:
        if rise_angle is None:
            raise ValueError("Необходимо задать rise_angle или t_start")
        if not (0 < rise_angle < np.pi/2):
            raise ValueError("rise_angle должен быть в интервале (0, π/2)")
        t_start = peak_time - (A / np.tan(rise_angle))

    if t_end is None:
        if fall_angle is None:
            raise ValueError("Необходимо задать fall_angle или t_end")
        if not (0 < fall_angle < np.pi/2):
            raise ValueError("fall_angle должен быть в интервале (0, π/2)")
        t_end = peak_time + (A / np.tan(fall_angle))

    # Проверка корректности временного интервала
    if t_start >= t_end:
        raise ValueError("t_start должно быть меньше t_end")
    if not (t_start < peak_time < t_end):
        raise ValueError("peak_time должно быть между t_start и t_end")

    # Генерация временной оси
    t = np.linspace(t_start, t_end, fs)
    triangular = np.zeros_like(t)

    # Расчет левого склона (наклон через тангенс угла)
    left_slope = A / (peak_time - t_start)
    left_mask = (t >= t_start) & (t <= peak_time)
    triangular[left_mask] = left_slope * (t[left_mask] - t_start)

    # Расчет правого склона (наклон через тангенс угла)
    right_slope = A / (t_end - peak_time)
    right_mask = (t > peak_time) & (t <= t_end)
    triangular[right_mask] = A - right_slope * (t[right_mask] - peak_time)

    return t, triangular

def generate_barker_code(length,amplitude = 1,frequency = 100,phase = 0,sample_rate = 10000,t_start=None, t_end=None):
    """
    Генерирует последовательность кода Баркера для заданной длины.

    Поддерживаемые длины: 2, 3, 4, 5, 7, 11, 13.

    Аргументы:
        length (int): Длина требуемого кода Баркера

    Вызывает:
        ValueError: Если запрошена неподдерживаемая длина

    # Параметры сигнала

    amplitude = 1      # Амплитуда (Вольты)
    frequency = 100       # Частота (Герцы)
    phase = 0            # Начальная фаза (радианы)
    sample_rate = 1000  # Частота дискретизации (Гц)   

    """
    barker_sequences = {
        2: [1, -1],
        3: [1, 1, -1],
        4: [1, 1, 1, -1],
        5: [1, 1, 1, -1, 1],
        7: [1, 1, 1, -1, -1, 1, -1],
        11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
    }

    if length not in barker_sequences:
        raise ValueError(
            f"Неподдерживаемая длина кода Баркера: {length}. "
            f"Допустимые длины: {list(barker_sequences.keys())}"
        )

        # Определение временного диапазона
    period = 1/frequency 
    duration = period*length

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    harmonic1 = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    harmonic2 = amplitude * np.sin(2 * np.pi * frequency * t + (phase + np.pi))

    signal = harmonic1
    i = 0
    for start_time in np.arange(0, duration, period):
            end_time = start_time + period
            if barker_sequences[length][i] == 1:
                signal[(t >= start_time) & (t < end_time)] = harmonic2[(t >= start_time) & (t < end_time)]
            i+=1
    return t,signal


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    c = 11 
    t,signal = generate_barker_code(c)
    plt.plot(t, signal)  
    plt.title(f'Код Баркера {c}')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()
