 %generationSignalWithNegativeFrequencies.m
% Генерация кода Баркера, гармонического сигнала и построение спектров с отрицательными частотами
clear; close all; clc;

% Параметры
c = 13;              % Длина кода Баркера
fs = 10000;          % Частота дискретизации, Гц
freq = 100;          % Частота гармонического сигнала, Гц
period = 1/freq;     % Период сигнала
duration = period * c; % Общая длительность сигнала

% Временная ось
t = linspace(0, duration, floor(fs*duration));

% --- Генерация кода Баркера вручную ---
cb3 = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1];
sig = zeros(size(t));
period_sample_count = floor(length(t)/c);

for i = 1:c
    start_idx = (i-1)*period_sample_count + 1;
    end_idx = i*period_sample_count;
    if end_idx > length(t)
        end_idx = length(t);
    end
    sig(start_idx:end_idx) = cb3(i);
end

% --- Построение графика кода Баркера ---
figure('Position',[100 100 1200 600]);
subplot(2,2,1);
plot(t, sig, 'LineWidth', 1.5);
title(['Код Баркера ', num2str(c)]);
xlabel('Время (с)');
ylabel('Амплитуда');
grid on;

% --- Вычисление и построение спектра кода Баркера с отрицательными частотами ---
subplot(2,2,2);
plotSpectrumWithNegativeFrequencies(sig, fs);
title(['Спектр кода Баркера ', num2str(c)]);

% --- Генерация гармонического сигнала, модулированного кодом Баркера ---
t2 = linspace(0, duration, floor(fs*duration));
signal = zeros(size(t2));

for i = 1:c
    start_idx = (i-1)*period_sample_count + 1;
    end_idx = i*period_sample_count;
    if end_idx > length(t2)
        end_idx = length(t2);
    end
    signal(start_idx:end_idx) = cb3(i) * sin(2*pi*freq*t2(start_idx:end_idx));
end

% --- Построение графика гармонического сигнала ---
subplot(2,2,3);
plot(t2, signal, 'LineWidth', 1.5);
title(['Гармонический сигнал, модулированный кодом Баркера ', num2str(c)]);
xlabel('Время (с)');
ylabel('Амплитуда');
grid on;

% --- Вычисление и построение спектра гармонического сигнала с отрицательными частотами ---
subplot(2,2,4);
plotSpectrumWithNegativeFrequencies(signal, fs);
title('Амплитудный спектр гармонического сигнала');

% --- Функция вычисления и построения спектра с отрицательными частотами ---
function plotSpectrumWithNegativeFrequencies(signal, fs)
    n = length(signal);
    % Вычисляем FFT
    fft_result = fft(signal);
    % Нормируем амплитуду
    amplitudes = abs(fft_result) / n;
    % Центрируем спектр (нулевая частота посередине)
    amplitudes = fftshift(amplitudes);
    % Формируем ось частот от -fs/2 до fs/2
    if mod(n,2) == 0
        f = (-n/2:n/2-1)*(fs/n);
    else
        f = (-(n-1)/2:(n-1)/2)*(fs/n);
    end
    frequencies = f;
    % Строим спектр
    plot(frequencies, amplitudes, 'LineWidth', 1.5);
    xlabel('Частота (Гц)');
    ylabel('Амплитуда');
    xlim([-fs/2 fs/2]);
    grid on;
end
