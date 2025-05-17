function generationSignal
% generationSignal - демонстрация генерации сигналов и кода Баркера
%
% Для использования функций вызывайте их из командной строки или другого скрипта:
% [t, g] = gaussian_pulse(0,1,1,[],[],1000);
% [t, r] = rectangular_pulse(0,1,1,[],[],1000);
% [t, tr] = triangular_pulse(0, pi/6, pi/6, 1, [], [], 1000);
% [t, b] = generate_barker_code(11,0,1,100,0,10000);

% Пример построения кода Баркера длины 11
c = 11;
[t, signal] = generate_barker_code(c);
figure;
plot(t, signal);
title(['Код Баркера ', num2str(c)]);
xlabel('Время (с)');
ylabel('Амплитуда');
grid on;

end

function [t, gaussian] = gaussian_pulse(mu, sigma, A, t_start, t_end, fs)
% gaussian_pulse - генерация гауссовского импульса
% Параметры:
%   mu - центр импульса (по умолчанию 0)
%   sigma - стандартное отклонение (ширина)
%   A - амплитуда
%   t_start, t_end - временной интервал (если пустые, по умолчанию mu ± 3*sigma)
%   fs - частота дискретизации (по умолчанию 1000)

if nargin < 1 || isempty(mu), mu = 0; end
if nargin < 2 || isempty(sigma), sigma = 1; end
if sigma <= 0
    error('sigma должно быть положительным.');
end
if nargin < 3 || isempty(A), A = 1; end
if nargin < 4 || isempty(t_start), t_start = mu - 3*sigma; end
if nargin < 5 || isempty(t_end), t_end = mu + 3*sigma; end
if nargin < 6 || isempty(fs), fs = 1000; end

t = linspace(t_start, t_end, floor(fs*(t_end - t_start)));
gaussian = A * exp(-((t - mu).^2) / (2*sigma^2));
end

function [t, pulse] = rectangular_pulse(start_time, duration, A, t_start, t_end, fs)
% rectangular_pulse - генерация прямоугольного импульса
% Параметры:
%   start_time - начало импульса
%   duration - длительность импульса
%   A - амплитуда
%   t_start, t_end - временной интервал (если пустые, по умолчанию start_time ± 10*duration)
%   fs - частота дискретизации

if nargin < 1 || isempty(start_time), start_time = 0; end
if nargin < 2 || isempty(duration), duration = 1; end
if nargin < 3 || isempty(A), A = 1; end
if nargin < 4 || isempty(t_start), t_start = start_time - 10*duration; end
if nargin < 5 || isempty(t_end), t_end = start_time + 10*duration; end
if nargin < 6 || isempty(fs), fs = 1000; end

t = linspace(t_start, t_end, floor((t_end - t_start)*fs));
pulse = zeros(size(t));
pulse(t >= start_time & t <= (start_time + duration)) = A;
end

function [t, triangular] = triangular_pulse(peak_time, rise_angle, fall_angle, A, t_start, t_end, fs)
% triangular_pulse - генерация треугольного импульса с углами наклона в радианах
% Параметры:
%   peak_time - время максимума
%   rise_angle - угол наклона левого склона (0 < angle < pi/2)
%   fall_angle - угол наклона правого склона (0 < angle < pi/2)
%   A - амплитуда
%   t_start, t_end - временные границы (если пустые, вычисляются по углам)
%   fs - частота дискретизации

if nargin < 1 || isempty(peak_time), peak_time = 0; end
if nargin < 2 || isempty(rise_angle)
    error('Необходимо задать rise_angle или t_start');
end
if nargin < 3 || isempty(fall_angle)
    error('Необходимо задать fall_angle или t_end');
end
if nargin < 4 || isempty(A), A = 1; end
if nargin < 7 || isempty(fs), fs = 1000; end

if isempty(t_start)
    if ~(rise_angle > 0 && rise_angle < pi/2)
        error('rise_angle должен быть в интервале (0, pi/2)');
    end
    t_start = peak_time - A / tan(rise_angle);
end
if isempty(t_end)
    if ~(fall_angle > 0 && fall_angle < pi/2)
        error('fall_angle должен быть в интервале (0, pi/2)');
    end
    t_end = peak_time + A / tan(fall_angle);
end

if t_start >= t_end
    error('t_start должно быть меньше t_end');
end
if ~(t_start < peak_time && peak_time < t_end)
    error('peak_time должно быть между t_start и t_end');
end

t = linspace(t_start, t_end, fs);
triangular = zeros(size(t));

left_slope = A / (peak_time - t_start);
right_slope = A / (t_end - peak_time);

left_mask = (t >= t_start) & (t <= peak_time);
triangular(left_mask) = left_slope * (t(left_mask) - t_start);

right_mask = (t > peak_time) & (t <= t_end);
triangular(right_mask) = A - right_slope * (t(right_mask) - peak_time);
end

function [t, signal] = generate_barker_code(length_code, start_time, amplitude, frequency, phase, sample_rate, t_start, t_end)
% generate_barker_code - генерация сигнала с кодом Баркера заданной длины
% Поддерживаемые длины: 2,3,4,5,7,11,13
%
% Параметры:
%   length_code - длина кода Баркера
%   start_time - время начала сигнала
%   amplitude - амплитуда
%   frequency - частота несущей
%   phase - начальная фаза
%   sample_rate - частота дискретизации
%   t_start, t_end - временной интервал (по умолчанию 0 и duration)
%
% Возвращает:
%   t - временная ось
%   signal - сигнал с кодом Баркера

if nargin < 1, error('Требуется длина кода Баркера'); end
if nargin < 2 || isempty(start_time), start_time = 0; end
if nargin < 3 || isempty(amplitude), amplitude = 1; end
if nargin < 4 || isempty(frequency), frequency = 100; end
if nargin < 5 || isempty(phase), phase = 0; end
if nargin < 6 || isempty(sample_rate), sample_rate = 10000; end

barker_sequences = containers.Map( ...
    {2,3,4,5,7,11,13}, ...
    {[1,-1],[1,1,-1],[1,1,1,-1],[1,1,1,-1,1],[1,1,1,-1,-1,1,-1], ...
    [1,1,1,-1,-1,-1,1,-1,-1,1,-1],[1,1,1,1,1,-1,-1,1,1,-1,1,-1,1]});

if ~isKey(barker_sequences, length_code)
    error(['Неподдерживаемая длина кода Баркера: ', num2str(length_code), ...
        '. Допустимые длины: 2,3,4,5,7,11,13']);
end
 
sequence = barker_sequences(length_code);
period = 1 / frequency;
duration = period * length_code;

if nargin < 7 || isempty(t_start), t_start = 0; end
if nargin < 8 || isempty(t_end), t_end = duration; end

t = linspace(0, duration, floor(sample_rate * duration));

harmonic1 = amplitude * sin(2*pi*frequency*t + phase);
harmonic2 = amplitude * sin(2*pi*frequency*t + phase + pi);

signal = harmonic1;

for i = 0:length_code-1
    start_seg = i * period;
    end_seg = start_seg + period;
    idx = (t >= start_seg) & (t < end_seg);
    if sequence(i+1) == 1
        signal(idx) = harmonic2(idx);
    end
end

end
