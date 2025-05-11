import numpy as np
from scipy.fft import fft, fftfreq, fftshift

def compute_spectrum(signal, sampling_rate=1.0):
    """
    Параметры:
    signal :  Входной одномерный сигнал.
    sampling_rate :  Частота дискретизации сигнала (в Гц). По умолчанию 1.0.

    Возвращает:
    frequencies : ndarray
        Массив частот (в Гц).
    amplitudes : ndarray
        Массив амплитуд соответствующих частот.
    """    
    n = len(signal)
    fft_result = fft(signal)
    amplitudes = np.abs(fft_result) / n
    frequencies = fftshift(fftfreq(n, d=1/sampling_rate))
    amplitudes = fftshift(amplitudes)
    return frequencies, amplitudes
