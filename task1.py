import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from getSpectrum import compute_spectrum
from generationSignal import gaussian_pulse, generate_barker_code

t = np.linspace(0, 1, int(1e6), endpoint=False)

def generate_rectangular_pulse(t, duration):
    return np.where((t >= 0) & (t <= duration), 1.0, 0.0)

def generate_rectangular_signal(
    frequency: float,
    duty_cycle: float = 0.5,
    amplitude: float = 1.0,
    duration: float = 0.001,  # 1 мс по умолчанию
    sampling_rate: float = 10e6  # 10 МГц по умолчанию
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0, duration, 1 / sampling_rate)
    
    period = 1 / frequency
    
    signal = amplitude * ((t % period) < duty_cycle * period).astype(float)
    
    return t, signal

rectangular_pulse100 = generate_rectangular_pulse(t,100)
rectangular_pulse01 = generate_rectangular_pulse(t,0.0001)

plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
plt.title("Прямоугольного импульс длительностью 100 сек")
plt.grid(True)
plt.plot(t,rectangular_pulse100 )


plt.subplot(2,2,2)
plt.title("Спектр прямоугольного импульса длительностью 100 сек")
plt.grid(True)
plt.plot(*compute_spectrum(rectangular_pulse100,100))

plt.subplot(2,2,3)
plt.title("Прямоугольного импульса длительностью 100 мкс")
plt.grid(True)
plt.plot(t[:1000],rectangular_pulse01[:1000])

plt.subplot(2,2,4)
plt.title("Спектр прямоугольного импульса длительностью 100 мкс")
plt.grid(True)
plt.plot(*compute_spectrum(rectangular_pulse01,30e6))

plt.figure(figsize=(20,10))

fs = 300e3
t,signal = generate_rectangular_signal(
    frequency=fs,
    amplitude=1.0,
    duration=0.0001,  # 100 мкс
    sampling_rate=30e6  # 30 МГц
    )

# Визуализация первых 10 периодов
n_periods_to_plot = 100
samples_per_period = int(1 / fs *30e6)
samples_to_plot = n_periods_to_plot * samples_per_period

plt.subplot(1,2,1)
plt.title("Прямоугольный сигнал с частотой 300кгц длительностью 100мкс")
plt.grid(True)
plt.plot(t[:samples_to_plot], signal[:samples_to_plot])

plt.subplot(1,2,2)
plt.title("Спектр прямоугольного сигнал с частотой 300кгц")
plt.grid(True)
plt.plot(*compute_spectrum(signal,fs))

plt.figure(figsize=(20,10))

def generate_signal(freq, duration, sampling_rate):
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

# Параметры сигналов
fs = 1000  # Частота дискретизации (Гц)
duration = 1.0  # Длительность (сек)

t1, signal1 = generate_signal(freq=50, duration=duration, sampling_rate=fs)

t2, signal2 = generate_signal(freq=25, duration=duration, sampling_rate=fs)

result_signal = signal1 * signal2 


plt.subplot(3, 2, 1)
plt.plot(t1, signal1)
plt.grid(True)
plt.title("Сигнал 1: 50 Гц")

plt.subplot(3, 2, 3)
plt.plot(t2, signal2)
plt.grid(True)
plt.title("Сигнал 2: 25 Гц")

plt.subplot(3, 2, 5)
plt.plot(t1, result_signal)
plt.grid(True)
plt.title("Результат умножения сигналов")

plt.subplot(3, 2, 2)
plt.title("Спектр cигнала 1: 50 Гц")
plt.grid(True)
plt.stem(*compute_spectrum(signal1,fs))
plt.xlim(-fs/10, fs/10)

plt.subplot(3, 2, 4)
plt.title("Спектр cигнала 2: 20 Гц")
plt.grid(True)
plt.xlim(-fs/10, fs/10)
plt.stem(*compute_spectrum(signal2,fs))

plt.subplot(3, 2, 6)
plt.title("Спектр умножения сигналов")
plt.grid(True)
plt.xlim(-fs/10, fs/10)
plt.stem(*compute_spectrum(result_signal,fs))

plt.figure(figsize=(20,10))
from scipy.fft import fft, fftfreq, fftshift

def generate_gaussian_pulse(fs, duration, sigma):
    """
    Генерирует гауссовский импульс.

    Параметры:
    fs (int): Частота дискретизации (Гц).
    duration (float): Длительность сигнала (сек).
    sigma (float): Стандартное отклонение (ширина импульса, сек).

    Возвращает:
    t (ndarray): Временная шкала.
    pulse (ndarray): Значения импульса.
    """
    t = np.linspace(-duration/2, duration/2, int(fs * duration), endpoint=False)
    pulse = np.exp(-t**2 / (2 * sigma**2))
    return t, pulse

# Параметры сигнала
fs = 4000       # Частота дискретизации
duration = 3.0  # Длительность в секундах
sigma = 0.1     # Ширина импульса

# Генерация импульса
t, pulse = generate_gaussian_pulse(fs, duration, sigma)

# Расчет спектра
N = len(pulse)
fft_values = fft(pulse)
fft_amplitude = np.abs(fft_values) / N  # Нормировка амплитуды
freq = fftfreq(N, 1/fs)
freq = fftshift(freq)
fft_amplitude = fftshift(fft_amplitude)

plt.subplot(2, 1, 1)
plt.plot(t, pulse)
plt.title('Гауссовский импульс')
plt.xlabel('Время (с)')
plt.grid(True)

# График спектра
plt.subplot(2, 1, 2)
plt.plot(*compute_spectrum(pulse,fs))
plt.title('Амплитудный спектр')
plt.xlim(-20,20)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid(True)

plt.figure(figsize=(20,10)) 


def generate_periodic_gaussian_pulse(fs, duration, sigma, period):
    """Генерация периодического гауссовского импульса"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    pulse = np.zeros_like(t)
    num_pulses = int(duration / period)
    
    for n in range(num_pulses):
        center = n * period
        pulse += np.exp(-(t - center)**2 / (2 * sigma**2))
    
    return t, pulse

# Параметры сигнала
fs = 5000       # Высокая частота дискретизации
duration = 5.0  # Увеличенная длительность
sigma = 0.05    # Ширина импульса
period = 0.5    # Период повторения (2 Гц)

# Генерация сигнала
t, pulse = generate_periodic_gaussian_pulse(fs, duration, sigma, period)

# Применение оконной функции и zero-padding
window = np.hanning(len(pulse))
pulse_windowed = pulse * window
N_fft = 10 * len(pulse)  # Увеличение точек FFT

# Расчет спектра
fft_values = fft(pulse_windowed, n=N_fft)
fft_amplitude = np.abs(fft_values) / len(pulse)  # Линейная нормировка
freq = fftfreq(N_fft, 1/fs)
freq = fftshift(freq)
fft_amplitude = fftshift(fft_amplitude)

plt.subplot(3, 1, 1)
plt.plot(t, pulse)
plt.title('Периодический гауссовский импульс')
plt.grid(True)

# Спектр в логарифмическом масштабе
plt.subplot(3, 1, 2)
plt.plot(freq, 20 * np.log10(fft_amplitude + 1e-10))  # dB scale
plt.title('Амплитудный спектр (в dB)')
plt.ylabel('Амплитуда (dB)')
plt.xlim(-20, 20)  # Увеличим диапазон для видимости гармоник
plt.ylim(-160, 0) # Диапазон dB для наглядности
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(freq, fft_amplitude)
plt.title('Амплитудный спектр (линейный масштаб)')
plt.ylabel('Амплитуда')
plt.xlim(-15, 15)    # Диапазон частот до 50 Гц
plt.ylim(-0.01, 0.15)  # Подбираем диапазон для видимости пиков
plt.grid(True)
plt.figure(figsize=(20,10))
# ================== Параметры сигнала ==================
fs = 1000           # Частота дискретизации (Гц)
t_total = 3         # Общая длительность (с)
fc = 10             # Несущая частота (Гц)
dev = 20             # Девиация частоты для FM (Гц)

# ================== Генерация сигнала ==================
t = np.arange(0, t_total, 1/fs)
signal = np.where(((t >= 0.5) & (t <= 1.0)) | ((t >= 1.5) & (t <= 2.0)), 1, 0)

# ================== Модуляции ==================
# Амплитудная модуляция (AM)
am_signal = signal * np.sin(2 * np.pi * fc * t)

# Частотная модуляция (FM)
integral = np.cumsum(signal) * (1/fs)
fm_signal = np.cos(2 * np.pi * (fc * t + dev * integral))

# ================== Визуализация ==================

def plot_spectrum(ax, signal, fs, title):
    n = len(signal)
    f = np.fft.fftfreq(n, 1/fs)
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft) * 2 / n  # Линейная амплитуда

    ax.plot(f, magnitude)
    ax.set_title(title)
    ax.set_xlabel('Частота (Гц)')
    ax.set_ylabel('Амплитуда')
    ax.grid(True)
    ax.set_xlim(-50, 50)  # Фиксированный диапазон частот
    ax.set_ylim(0, magnitude.max()*1.1)  # Автоподбор по Y

# Временные графики
plt.subplot(3, 2, 1)
plt.plot(t, signal)
plt.title('Исходный сигнал')
plt.grid(True)
plt.ylim(-0.2, 1.2)

plt.subplot(3, 2, 3)
plt.plot(t, am_signal)
plt.title('AM сигнал')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(t, fm_signal)
plt.title('FM сигнал')
plt.grid(True)

# Спектры
ax_spec1 = plt.subplot(3, 2, 2)
plot_spectrum(ax_spec1, signal, fs, 'Спектр исходного сигнала')

ax_spec2 = plt.subplot(3, 2, 4)
plot_spectrum(ax_spec2, am_signal, fs, 'Спектр AM')

ax_spec3 = plt.subplot(3, 2, 6)
plot_spectrum(ax_spec3, fm_signal, fs, 'Спектр FM')
plt.tight_layout()

plt.show()
