import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from getSpectrum import compute_spectrum
from generationSignal import gaussian_pulse, generate_barker_code
plt.figure(figsize=(20,10))
cb3 = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
c = 13 
plt.subplot(2, 2, 1)

fs = 10000
period = 1/100
duration = period*c

t = np.linspace(0, duration, int(fs*duration), endpoint=False)
sig = np.zeros_like(t)
period = duration/c
i = 0
for start_time in np.arange(0, duration, period):
    try:
        end_time = start_time + period
        sig[(t >= start_time) & (t < end_time)] = cb3[i]
        i+=1
    except:
        pass

plt.subplot(2, 2, 1)
plt.plot(t, sig)
plt.title(f'Код Баркера {c}')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(*compute_spectrum(sig,10000))
plt.title(f'Спектр кода Баркера {c}')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.xlim(-fs/5,fs/5)
plt.grid(True)

plt.subplot(2, 2, 3)
t,signal = generate_barker_code(c)
plt.plot(t, signal)
plt.title(f'Гармонический сигнал модулируемый Кодом Баркера {c}')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(*compute_spectrum(signal,10000))
plt.title('Амплитудный спектр')
plt.xlim(-fs/5,fs/5)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid(True)

plt.show()
