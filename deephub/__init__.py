import warnings

# Suppress warning errors from importing pandas
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd  # noqa: F401  see: https://github.com/ContinuumIO/anaconda-issues/issues/6678
    import h5py  # noqa: F401
    import scipy  # noqa: F401
    from scipy import ndimage  # noqa: F401
    import sklearn  # noqa: F401
    from sklearn.utils import extmath  # noqa: F401
    from sklearn import metrics  # noqa: F401

__version__ = '0.9.0'
