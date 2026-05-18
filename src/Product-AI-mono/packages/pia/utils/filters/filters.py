from collections import deque
from statistics import mean

import numpy as np
from scipy.signal import savgol_filter

from .base import FilterBase, FilterConfigBase


class AverageFilter(FilterBase):
    def __init__(self, config: FilterConfigBase) -> None:
        super().__init__(config)
        self.k = 0
        self.config = config

    def filter(self, obs: float) -> float:
        self.k += 1
        alpha = (self.k - 1) / self.k
        avg = alpha * self.pre_v + (1 - alpha) * obs
        return avg


class MovingAverageFilter(FilterBase):
    def __init__(self, config: FilterConfigBase) -> None:
        self.window = deque(maxlen=config.window_length)
        super().__init__(config)

    def filter(self, obs: float) -> float:
        self.window.append(obs)
        avg = mean(self.window)
        if avg == 0:
            avg = 1
        return sum(self.window) / avg


class StreamingSavitzkyGolayFilter(FilterBase):
    def __init__(self, config: FilterConfigBase):
        super().__init__(config)
        """
        Initializes the Savitzky-Golay filter for streaming data with specific window length and polynomial order.

        Args:
        window_length (int): The length of the filter window. Must be a positive odd integer.
        polyorder (int): The order of the polynomial used to fit the samples. Must be less than window_length.
        """
        window_length = config.window_length
        polyorder = config.keyword["polyorder"]
        if window_length % 2 == 0 or window_length < 1:
            raise ValueError("Window length must be a positive odd integer")
        if polyorder >= window_length:
            raise ValueError("Polyorder must be less than window length")

        self.window_length = window_length
        self.polyorder = polyorder
        self.data_stream = deque(
            maxlen=window_length
        )  # Use deque to handle a sliding window of data

    def filter(self, new_value: float) -> float:
        """
        Updates the data stream with a new value and applies the Savitzky-Golay filter to the current window.

        Args:
        new_value (float): The new data point to add to the stream.

        Returns:
        float: The filtered value at the current position (if enough points have been collected).
        """
        self.data_stream.append(new_value)
        if len(self.data_stream) == self.window_length:
            # Apply the filter only if we have enough data points
            filtered_data = savgol_filter(
                list(self.data_stream), self.window_length, self.polyorder
            )
            return filtered_data[-1]  # Return the latest filtered point
        return new_value  # Not enough data to apply the filter


class EWMAFilter(FilterBase):
    def __init__(self, config: FilterConfigBase) -> None:
        super().__init__(config)
        """
        Initializes the Exponentially Weighted Moving Average filter with a smoothing factor.

        Args:
        alpha (float): Smoothing factor for EWMA, between 0 and 1. Higher values give more weight to recent data.
        """
        self.alpha = config.alpha_weight
        self.average = None  # Initial average is not set

    def filter(self, new_value: float) -> float:
        """
        Updates the EWMA with a new value and returns the current moving average.

        Args:
        new_value (float): The new data point to be added.

        Returns:
        float: The updated exponentially weighted moving average.
        """
        if self.average is None:
            self.average = new_value  # First value initialization
        else:
            self.average = self.alpha * new_value + (1 - self.alpha) * self.average

        return self.average


class MADFilter(FilterBase):
    def __init__(self, config: FilterConfigBase):
        """
        Initializes the Median Absolute Deviation filter with a specific window length.

        Args:
        window_length (int): The number of data points to use for calculating the median and MAD.
        """
        window_length = config.window_length
        self.window_length = window_length
        self.data_stream = deque(maxlen=window_length)

    def filter(self, new_value: float) -> float:
        """
        Updates the MAD with a new value, computes the median and MAD, and identifies outliers.

        Args:
        new_value (float): The new data point to be added.

        Returns:
        tuple: A boolean indicating if the new value is an outlier, and the median and MAD values.
        """
        self.data_stream.append(new_value)
        if len(self.data_stream) == self.window_length:
            median = np.median(self.data_stream)
            deviations = np.abs(np.array(self.data_stream) - median)
            mad = np.median(deviations)
            if mad == 0:  # To avoid division by zero in case all numbers are the same
                mad = 1e-6
            modified_z_score = 0.6745 * (new_value - median) / mad
            return modified_z_score  # Common threshold for outlier detection
        return 0
