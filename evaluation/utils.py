import numpy as np

def _next_power_of_2(x):
    """Calculate the next power of 2 no smaller than x."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)