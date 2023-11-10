import matplotlib.pyplot as plt
import numpy as np
import sys


class SignalGenerator:
    def __init__(self):
        self.phase_min, self.phase_max = (0, 360)
        self.samples_per_period = 1000
        self.duration_min, self.duration_max = (0.001, 0.1)
        self.dc_factor = 2
        self.duty_min, self.duty_max = (0.01, 0.99)
        self.pwl_error = 1
        self.max_samples = 1e6
        self.amplitude_ranges = [
            (1e-6, 1e-5),
            (1e-5, 1e-4),
            (1e-4, 1e-3),
            (1e-3, 1e-2),
            (1e-2, 1e-1),
            (1e-1, 1),
            (1, 1e1),
        ]
        self.frequency_ranges = [(1, 1e1), (1e1, 1e2), (1e2, 1e3), (1e3, 1e4)]

    def linearize(self, signal, time):
        if len(signal) == 0:
            return [(0, 0), (0, 0)]

        # find the point between (t_0, s_0) and (t_n, s_n)
        def line(t):
            return (s_n - s_0) / (t_n - t_0) * (t - t_0) + s_0

        # find the max and min values of the signal
        max_value_time = np.argmax(signal)
        min_value_time = np.argmin(signal)
        max_value = signal[max_value_time]
        min_value = signal[min_value_time]

        t_0 = time[0]
        t_n = time[-1]
        s_0 = signal[0]
        s_n = signal[-1]

        if len(signal) == 2 or max_value == min_value:
            # return the line
            return [(time[0], signal[0]), (time[-1], signal[-1])]

        if max_value > s_0 and max_value > s_n:
            # split the signal at the max value
            left_signal = signal[:max_value_time]
            left_time = time[:max_value_time]
            right_signal = signal[max_value_time:]
            right_time = time[max_value_time:]
            left_segments = self.linearize(left_signal, left_time)
            right_segments = self.linearize(right_signal, right_time)
            return left_segments + right_segments

        if min_value < s_0 and min_value < s_n:
            # split the signal at the min value
            left_signal = signal[:min_value_time]
            left_time = time[:min_value_time]
            right_signal = signal[min_value_time:]
            right_time = time[min_value_time:]
            left_segments = self.linearize(left_signal, left_time)
            right_segments = self.linearize(right_signal, right_time)
            return left_segments + right_segments

        # find the error between the line and the signal at each point
        total_error = 0
        max_error = 0
        max_error_time = 0

        for t in range(len(time)):
            t_step = time[t]
            err = abs(line(t_step) - signal[t])
            if err > max_error:
                max_error = err
                max_error_time = t
            total_error += err
        norm_error = total_error / abs(max_value - min_value)

        if norm_error > self.pwl_error:
            # evenly split the signal
            left_signal = signal[:max_error_time]
            left_time = time[:max_error_time]
            right_signal = signal[max_error_time:]
            right_time = time[max_error_time:]
            left_segments = self.linearize(left_signal, left_time)
            right_segments = self.linearize(right_signal, right_time)
            return left_segments + right_segments

        # else return the line
        return [(t_0, s_0), (t_n, s_n)]

    def random_value(self, ranges):
        uniform_distributions = []
        for range in ranges:
            uniform_distributions.append(np.random.uniform(range[0], range[1]))
        return np.random.choice(uniform_distributions)

    def generate_sine(self):
        sys.setrecursionlimit(10000)
        amp = self.random_value(self.amplitude_ranges)
        freq = self.random_value(self.frequency_ranges)
        phase = np.random.uniform(self.phase_min, self.phase_max)
        dc = np.random.uniform(-self.dc_factor * amp, self.dc_factor * amp)
        period = 1 / freq
        duration = np.random.uniform(period, max(self.duration_max, period))
        num_periods = duration / period
        num_samples = int(num_periods * self.samples_per_period)
        times = np.linspace(0, duration, num_samples)
        angles = (2 * np.pi * freq * times) + (phase * np.pi / 180)
        signal = np.sin(angles)
        pwl_signal = self.linearize(signal, times)
        times, signal = zip(*pwl_signal)

        # preprocess for machine learning
        amp_coeff, amp_exp = self.to_ml_format(amp, -15, 12)
        mean_coeff, mean_exp = self.to_ml_format(dc, -15, 12)

        return {
            "signal": signal,
            "times": times,
            "mean": (mean_coeff, mean_exp),
            "amplitude": (amp_coeff, amp_exp),
            "frequency": freq,
            "phase": phase,
        }

    def to_ml_format(self, number, max_exponent):
        scientific_notation = format(number, "e")
        factor_str, exponent_str = scientific_notation.split("e")

        # Convert scientific notation to machine learning friendly format
        coefficient = float(factor_str) / 10
        exponent = int(exponent_str) + 1

        exponent_array = self.int_to_one_hot_array(exponent, max_exponent, True)

        return coefficient, exponent_array
    
    def int_to_one_hot_array(self, number, max_exponent, signed):
        length = max_exponent
        if signed:
            length += 1
        one_hot_array = np.zeros(length)
        if(number < 0):
            one_hot_array[0] = 1
            number = -number
        one_hot_array[number] = 1

    def scale_array(self, array: np.ndarray):
        max_value = np.max(array)
        # normalized_array = (array - mean) / max_value
        # mean_coefficient, mean_exponent = self.to_ml_format(mean)
        normalized_array = array / max_value
        max_coefficient, max_exponent = self.to_ml_format(max_value)

        return (
            normalized_array,
            # (mean_coefficient, mean_exponent),
            (max_coefficient, max_exponent),
        )

    def plot(self, signal, times):
        plt.plot(times, signal, linewidth=0.5, color="black")
        plt.scatter(times, signal, s=1, color="red")
        plt.show()

    def generate_pwm(self):
        amp = self.random_value(self.amplitude_ranges)
        freq = self.random_value(self.frequency_ranges)
        phase = np.random.uniform(self.phase_min, self.phase_max)
        duty = np.random.uniform(self.duty_min, self.duty_max)
        dc = np.random.uniform(-2 * amp, 2 * amp)
        duration = np.random.uniform(self.duration_min, self.duration_max)
        num_periods = duration * freq
        num_samples = int(num_periods * self.samples_per_period * duration)
        times = np.linspace(0, duration, num_samples)
        angles = 2 * np.pi * freq * times + phase
        signal = dc + np.where(np.mod(angles, 2 * np.pi) < 2 * np.pi * duty, amp, -amp)
        return {
            "signal_values": signal,
            "time_steps": times,
            "amplitude": amp,
            "frequency": freq,
            "phase": phase,
            "duty_cycle": duty,
        }