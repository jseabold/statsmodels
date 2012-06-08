"""
Seasonal Decomposition by Moving Averages
"""
import numpy as np
from scipy import signal

def decompose(x, typ="additive", filter=None, frequency=4):
    """
    Parameters
    ----------
    x : array-like
        Time series
    typ : str {"additive", "multiplicative"}
        Type of seasonal component. Abbreviations are accepted.
    filter : array-like
        The filter coefficients for filtering out the seasonal component.
        The default is a symmetric moving average.
    frequency : int
        Frequency of the series. Assumed to start at index 0.

    The additive model is Y[t] = T[t] + S[t] + e[t]

    The multiplicative model is Y[t] = T[t] * S[t] * e[t]

    """
    #NOTE: this would be a lot easier code if it just used pandas
    #TODO: replace with time series aware code. If not, period handling is
    #broken right?
    if frequency % 2 == 0: # split weights at ends
        filt = np.array([.5] + [1] * (frequency - 1) + [.5]) / frequency
    else:
        filt = np.array([1./frequency] * f)
    drop_idx = frequency // 2
    idx = np.arange(len(x))
    mask = ~((idx > drop_idx-1) & (idx < len(x) - drop_idx))
    ma_x = np.ma.masked_array(x, mask=mask)
    trend = np.ma.masked_array(np.convolve(filt, x, mode='same'), mask=mask)
    # this is faster than signal.lfilter(filt, [1], x) for size < 2**9
    # and both are faster than signal.fftconvolve
    if typ.startswith('a'):
        detrended = ma_x - trend # modify in place?
    elif typ.startsiwth('m'):
        detrended = ma_x / trend
    else:
        raise ValueError("typ %s not understood" % typ)

    idx = np.arange(drop_idx, len(detrended), frequency)
    period_averages = np.array([detrended[i::frequency].mean()
                            for i in range(frequency)])

    seasonal = np.tile(period_averages, len(x) // frequency)
    if typ.startswith('a'):
        period_averages -= np.mean(period_averages)
        resid = detrended - np.tile(period_averages,
                                                len(x) // frequency)
    else:
        period_averages /= np.mean(period_averages)
        resid = detrended / np.tile(period_averages,
                                                len(x) // frequency)

    trend[trend.mask] = np.nan
    resid[resid.mask] = np.nan
    detrended[detrended.mask] = np.nan
    # trend, seasonal, errors
    return seasonal, trend.data, resid.data



if __name__ == "__main__":
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809,
         530, 489, 540, 457, 195, 176, 337, 239, 128, 102, 232, 429, 3,
         98, 43, -141, -77, -13, 125, 361, -45, 184])
    seasonal, trend, resid = decompose(x)
