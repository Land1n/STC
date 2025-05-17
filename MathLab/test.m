% translated_signal_examples.m
clear; close all; clc;

%% --- Прямоугольные импульсы разной длительности ---
t = linspace(0, 1, 1e6);

rectangular_pulse100 = generate_rectangular_pulse(t, 100);
rectangular_pulse01 = generate_rectangular_pulse(t, 0.0001);

figure('Position',[100 100 1200 600]);
subplot(2,2,1)
plot(t, rectangular_pulse100, 'LineWidth',1.2);
title('Прямоугольный импульс длительностью 100 сек');
grid on;

subplot(2,2,2)
[fr, amp] = compute_spectrum(rectangular_pulse100, 100);
plot(fr, amp, 'LineWidth',1.2);
title('Спектр прямоугольного импульса длительностью 100 сек');
grid on;

subplot(2,2,3)
plot(t(1:1000), rectangular_pulse01(1:1000), 'LineWidth',1.2);
title('Прямоугольный импульс длительностью 100 мкс');
grid on;

subplot(2,2,4)
[fr, amp] = compute_spectrum(rectangular_pulse01, 30e6);
plot(fr, amp, 'LineWidth',1.2);
title('Спектр прямоугольного импульса длительностью 100 мкс');
grid on;

%% --- Прямоугольный сигнал с частотой 300 кГц ---
fs = 30e6;
[t2, signal] = generate_rectangular_signal(300e3, 0.5, 1.0, 0.0001, fs);

n_periods_to_plot = 100;
samples_per_period = round(1/300e3 * fs);
samples_to_plot = n_periods_to_plot * samples_per_period;

figure('Position',[100 100 1200 400]);
subplot(1,2,1)
plot(t2(1:samples_to_plot), signal(1:samples_to_plot), 'LineWidth',1.2);
title('Прямоугольный сигнал с частотой 300кГц, длительность 100мкс');
grid on;

subplot(1,2,2)
[fr, amp] = compute_spectrum(signal, fs);
plot(fr, amp, 'LineWidth',1.2);
title('Спектр прямоугольного сигнала с частотой 300кГц');
grid on;

%% --- Сигналы: умножение синусов и их спектры ---
fs = 1000;
duration = 1.0;

[t1, signal1] = generate_sine_signal(50, duration, fs);
[t2, signal2] = generate_sine_signal(25, duration, fs);
result_signal = signal1 .* signal2;

figure('Position',[100 100 1200 800]);
subplot(3,2,1)
plot(t1, signal1, 'LineWidth',1.2);
title('Сигнал 1: 50 Гц');
grid on;

subplot(3,2,3)
plot(t2, signal2, 'LineWidth',1.2);
title('Сигнал 2: 25 Гц');
grid on;

subplot(3,2,5)
plot(t1, result_signal, 'LineWidth',1.2);
title('Результат умножения сигналов');
grid on;

subplot(3,2,2)
[fr, amp] = compute_spectrum(signal1, fs);
stem(fr, amp, 'filled');
title('Спектр сигнала 1: 50 Гц');
xlim([-fs/10 fs/10]);
grid on;

subplot(3,2,4)
[fr, amp] = compute_spectrum(signal2, fs);
stem(fr, amp, 'filled');
title('Спектр сигнала 2: 25 Гц');
xlim([-fs/10 fs/10]);
grid on;

subplot(3,2,6)
[fr, amp] = compute_spectrum(result_signal, fs);
stem(fr, amp, 'filled');
title('Спектр умножения сигналов');
xlim([-fs/10 fs/10]);
grid on;

%% --- Гауссовский импульс и его спектр ---
fs = 4000;
duration = 3.0;
sigma = 0.1;
[t, pulse] = generate_gaussian_pulse(fs, duration, sigma);

figure('Position',[100 100 800 600]);
subplot(2,1,1)
plot(t, pulse, 'LineWidth',1.2);
title('Гауссовский импульс');
xlabel('Время (с)');
grid on;

subplot(2,1,2)
[fr, amp] = compute_spectrum(pulse, fs);
plot(fr, amp, 'LineWidth',1.2);
title('Амплитудный спектр');
xlim([-20 20]);
xlabel('Частота (Гц)');
ylabel('Амплитуда');
grid on;

%% --- Периодический гауссовский импульс и его спектр ---
fs = 5000;
duration = 5.0;
sigma = 0.05;
period = 0.5;
[t, pulse] = generate_periodic_gaussian_pulse(fs, duration, sigma, period);

window = hann(length(pulse))';
pulse_windowed = pulse .* window;
N_fft = 10 * length(pulse);

fft_values = fft(pulse_windowed, N_fft);
fft_amplitude = abs(fft_values) / length(pulse);
freq = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
fft_amplitude = fftshift(fft_amplitude);

figure('Position',[100 100 900 900]);
subplot(3,1,1)
plot(t, pulse, 'LineWidth',1.2);
title('Периодический гауссовский импульс');
grid on;

subplot(3,1,2)
plot(freq, 20*log10(fft_amplitude+1e-10), 'LineWidth',1.2);
title('Амплитудный спектр (в dB)');
ylabel('Амплитуда (dB)');
xlim([-20 20]);
ylim([-160 0]);
grid on;

subplot(3,1,3)
plot(freq, fft_amplitude, 'LineWidth',1.2);
title('Амплитудный спектр (линейный масштаб)');
ylabel('Амплитуда');
xlim([-15 15]);
ylim([-0.01 0.15]);
grid on;

%% --- AM и FM модуляция прямоугольного сигнала и их спектры ---
% Параметры
fs = 1000;          % Частота дискретизации, Гц
t_total = 3;        % Длительность сигнала, с
fc = 10;            % Частота несущей, Гц
freq_dev = 20;      % Девиация частоты для FM, Гц

% Временная ось
t = 0:1/fs:t_total-1/fs;

% Исходный прямоугольный сигнал (модулирующий)
mod_signal = double(((t >= 0.5) & (t <= 1)) | ((t >= 1.5) & (t <= 2)));

% AM модуляция: умножение на несущую
am_signal = mod_signal .* sin(2*pi*fc*t);

% FM модуляция: интеграл от модулирующего сигнала
integral_mod = cumsum(mod_signal) / fs;
fm_signal = cos(2*pi*fc*t + 2*pi*freq_dev*integral_mod);

% Построение сигналов
figure;
subplot(3,1,1);
plot(t, mod_signal, 'LineWidth',1.5);
title('Исходный прямоугольный сигнал');
xlabel('Время, с');
ylabel('Амплитуда');
grid on;

subplot(3,1,2);
plot(t, am_signal, 'LineWidth',1.5);
title('AM модулированный сигнал');
xlabel('Время, с');
ylabel('Амплитуда');
grid on;

subplot(3,1,3);
plot(t, fm_signal, 'LineWidth',1.5);
title('FM модулированный сигнал');
xlabel('Время, с');
ylabel('Амплитуда');
grid on;

% Функция для вычисления и построения спектра
function plotSpectrum(signal, fs, titleStr)
    n = length(signal);
    f = (-floor(n/2):ceil(n/2)-1)*(fs/n);
    S = fftshift(abs(fft(signal))/n);
    figure;
    plot(f, S, 'LineWidth',1.5);
    title(titleStr);
    xlabel('Частота, Гц');
    ylabel('Амплитуда');
    grid on;
    xlim([-fs/2 fs/2]);
end
plotSpectrum(mod_signal, fs, 'Спектр исходного прямоугольного сигнала');
plotSpectrum(am_signal, fs, 'Спектр AM сигнала');
plotSpectrum(fm_signal, fs, 'Спектр FM сигнала');
% Построение спектров



%% --- Локальные функции ---

function y = generate_rectangular_pulse(t, duration)
    y = double((t >= 0) & (t <= duration));
end

function [t, y] = generate_rectangular_signal(frequency, duty_cycle, amplitude, duration, sampling_rate)
    t = 0:1/sampling_rate:duration-1/sampling_rate;
    period = 1 / frequency;
    y = amplitude * double(mod(t, period) < duty_cycle * period);
end

function [f, amp] = compute_spectrum(signal, fs)
    n = length(signal);
    fft_result = fft(signal);
    amp = abs(fft_result)/n;
    amp = fftshift(amp);
    if mod(n,2)==0
        f = (-n/2:n/2-1)*(fs/n);
    else
        f = (-(n-1)/2:(n-1)/2)*(fs/n);
    end
end

function [t, y] = generate_sine_signal(freq, duration, fs)
    t = 0:1/fs:duration-1/fs;
    y = sin(2*pi*freq*t);
end

function [t, pulse] = generate_gaussian_pulse(fs, duration, sigma)
    t = linspace(-duration/2, duration/2, fs*duration);
    pulse = exp(-t.^2/(2*sigma^2));
end

function [t, pulse] = generate_periodic_gaussian_pulse(fs, duration, sigma, period)
    t = linspace(0, duration, fs*duration);
    pulse = zeros(size(t));
    num_pulses = floor(duration/period);
    for n = 0:num_pulses-1
        center = n * period;
        pulse = pulse + exp(-(t - center).^2/(2*sigma^2));
    end
end

function plot_spectrum_subplot(signal, fs, ttl)
    n = length(signal);
    f = (-n/2:n/2-1)*(fs/n);
    fft_val = fft(signal);
    mag = abs(fftshift(fft_val))*2/n;
    plot(f, mag, 'LineWidth',1.2);
    title(ttl);
    xlabel('Частота (Гц)');
    ylabel('Амплитуда');
    xlim([-50 50]);
    ylim([0 max(mag)*1.1]);
    grid on;
end
 
