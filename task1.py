import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from getSpectrum import compute_spectrum

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

plt.figure()
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

plt.figure()

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

plt.figure()

def generate_signal(freq, duration, sampling_rate):
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

# Параметры сигналов
fs = 1000  # Частота дискретизации (Гц)
duration = 1.0  # Длительность (сек)

t1, signal1 = generate_signal(freq=50, duration=duration, sampling_rate=fs)

t2, signal2 = generate_signal(freq=20, duration=duration, sampling_rate=fs)

result_signal = signal1 * signal2 


plt.subplot(3, 2, 1)
plt.plot(t1, signal1)
plt.grid(True)
plt.title("Сигнал 1: 50 Гц")

plt.subplot(3, 2, 3)
plt.plot(t2, signal2)
plt.grid(True)
plt.title("Сигнал 2: 20 Гц")

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

plt.show()

