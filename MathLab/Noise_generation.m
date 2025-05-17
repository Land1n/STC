function noisy_signal = add_awgn_noise(signal, snr_db)
% add_awgn_noise - добавляет аддитивный белый гауссовский шум (AWGN) к сигналу

% Входные параметры:
%   signal - входной сигнал (вектор)
%   snr_db - желаемое отношение сигнал/шум в децибелах
%
% Выходные параметры:
%   noisy_signal - сигнал с добавленным шумом (того же размера)

    signal_power = mean(abs(signal).^2);
    snr_linear = 10^(snr_db/10);
    noise_power = signal_power / snr_linear;

    noise = sqrt(noise_power) * randn(size(signal));
    noisy_signal = signal + noise;
end
