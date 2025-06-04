
import pytest
from drift_detector import calculate_psi, calculate_categorical_drift
import numpy as np
import pandas as pd

def test_psi_zero_for_identical_distributions():
    data = np.random.normal(0, 1, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.01

def test_categorical_drift_detection():
    expected = ['a'] * 50 + ['b'] * 50
    actual = ['a'] * 10 + ['b'] * 90
    score = 1 - calculate_categorical_drift(expected, actual)
    assert score > 0.2
