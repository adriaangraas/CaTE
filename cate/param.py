import warnings
from abc import ABC

import numpy as np


class Parameter(ABC):
    """A delayable value with optimization information."""

    def __init__(self, value, optimize: bool = True, bounds=None):
        """
        :param value: Can also be `Callable` to delay computation.
        :param optimize: Set to `False` to disable fitting procedure.
        :param bounds: `None` or a `tuple` of ndarray.
        """
        self._value = value
        self.optimize = optimize

        if bounds is None:
            lower = [-np.inf] * self.__len__()
            upper = [np.inf] * self.__len__()
            bounds = [lower, upper]

        self.bounds = bounds

    @property
    def value(self):
        if callable(self._value):
            return self._value()

        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        assert len(self) > 0

    def __len__(self):
        if np.isscalar(self._value):
            return 1

        return len(self._value)


class ScalarParameter(Parameter):
    def __init__(self, value, optimize: bool = True, bounds=None):
        if not np.isscalar(value):
            raise TypeError("`value` must be a scalar.")

        super(ScalarParameter, self).__init__(value, optimize, bounds)


class VectorParameter(Parameter):
    """A 3-vector ndarray that allows optimization.

    Here `optimize` has to be given explicitly because I can see use cases
    where points in the reconstruction volume are at known and unknown
    locations, so I don't want to implicitly choose one.
    """

    def __init__(self, value, optimize: bool, bounds=None):
        if issubclass(type(value), list):
            value = np.array(value, dtype=np.float)

        if not isinstance(value, np.ndarray):
            raise TypeError("`value` must be a `numpy.ndarray`.")

        if not len(value) == 3:
            raise ValueError("`np.ndarray` must have length 3.")

        super(VectorParameter, self).__init__(value, optimize, bounds)


def params2ndarray(params, optimizable_only=True, key='value'):
    """Packs a list of Packable into ndarray, and returns a list of types
    to restore to.

    :param params:
    :return:
    """
    # compute array length
    length = 0
    for p in params:
        if not issubclass(type(p), Parameter):
            warnings.warn("A value in `params` is not of type `Parameter`. "
                          "The value is ignored.", UserWarning)
            continue

        if optimizable_only and p.optimize is False:
            continue

        length += len(p)

    assert length > 0

    out = np.empty(length)
    idx = 0
    for p in params:
        if not issubclass(type(p), Parameter):
            continue

        if optimizable_only and p.optimize is False:
            continue

        len_p = len(p)
        if key == 'value':
            store = p.value
        elif key == 'min_bound':
            store = p.bounds[0]
        elif key == 'max_bound':
            store = p.bounds[1]
        else:
            return ValueError

        out[idx:idx + len_p] = store
        idx += len_p

    assert idx == length

    return out


def update_params(params, x: np.ndarray, optimizable_only=True):
    """In-place updating a list of parameters.

    Expect `params` and `x` to be given in the same order as that they were
    when they `params` was turned into an array.
    """
    idx = 0
    for p in params:
        if not issubclass(type(p), Parameter):
            continue

        if optimizable_only and p.optimize is False:
            continue

        len_p = len(p)
        assert len_p != 0
        p.value = x[idx: idx + len_p]
        idx += len_p

    assert idx == len(x)