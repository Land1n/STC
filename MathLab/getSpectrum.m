function [frequencies, amplitudes] = compute_spectrum(signal, sampling_rate)
% compute_spectrum - вычисляет спектр сигнала с центровкой нулевой частоты
%
% Входные параметры:
%   signal        - входной одномерный сигнал (вектор)
%   sampling_rate - частота дискретизации (Гц), по умолчанию 1.0
%
% Выходные параметры:
%   frequencies - вектор частот (Гц)
%   amplitudes  - амплитуды соответствующих частот

    if nargin < 2
        sampling_rate = 1.0;
    end

    n = length(signal);
    fft_result = fft(signal);
    amplitudes = abs(fft_result) / n;

    % Частоты с учетом дискретизации
    frequencies = (-floor(n/2) : ceil(n/2)-1) * (sampling_rate / n);

    % Сдвиг спектра, чтобы нулевая частота была в центре
    amplitudes = fftshift(amplitudes);
end
 
