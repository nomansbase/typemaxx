from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _as_float(
    name: str,
    value: object,
    non_negative: bool = False,
    strict_positive: bool = False,
) -> float:
    """
    Convert an input value to a finite Python float.

    Parameters
    ----------
    name : str
        Name of the parameter being validated. Used only to produce
        informative error messages.
    value : object
        Input object to convert to a float.
    non_negative: bool, optional
        Required to check for non-negativity. Default is
        'False'.
    strict_positive: bool, optional
        Required to check for strict positivity. If active,
        silences the non_negative flag above. Default is 'False'.

    Returns
    -------
    float
        The converted finite floating-point value.

    Raises
    ------
    TypeError
        If `value` cannot be converted to a float.
    ValueError
        If the converted value is not finite (for example, ``NaN``,
        ``+inf``, or ``-inf``).

    Notes
    -----
    This function is intended as a small validation helper for scalar
    numeric parameters.
    """
    try:
        out = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be float-like. Found {type(value)}.") from e

    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite. Found {out}.")

    if strict_positive:
        if value <= 0:
            raise ValueError(f"{name} must be strictly positive. Found: {value}.")
    elif non_negative:
        if value < 0:
            raise ValueError(f"{name} must be non-negative. Found: {value}.")

    return out


def _as_int(name: str, value: object, non_negative: int = False, strict_positive=False) -> int:
    """
    Convert an input value to a Python integer.

    Parameters
    ----------
    name : str
        Name of the parameter being validated. Used only to produce
        informative error messages.
    value : object
        Input object to convert to an integer.
    non_negative: bool, optional
        Required to check for non-negativity. Default is
        'False'.
    strict_positive: bool, optional
        Required to check for strict positivity. If active,
        silences the non_negative flag above. Default is 'False'.

    Returns
    -------
    int
        The converted integer value.

    Raises
    ------
    TypeError
        If `value` cannot be converted to an integer.

    Notes
    -----
    This function does not check bounds, sign, or semantic validity.
    It only enforces integer convertibility.
    """
    try:
        out = int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be int-like. Found {type(value)}.") from e
    if strict_positive:
        if value <= 0:
            raise ValueError(f"{name} must be strictly positive. Found: {value}.")
    elif non_negative:
        if value < 0:
            raise ValueError(f"{name} must be non-negative. Found: {value}.")
    return out


def _as_list(name: str, value: object, allow_empty: bool = True) -> list:
    """
    Convert a list-like object to a Python list.

    Parameters
    ----------
    name : str
        Name of the parameter being validated. Used only to produce
        informative error messages.
    value : object
        Input object expected to be list-like. Supported types are
        `list`, `tuple`, and `numpy.ndarray`.
    allow_empty: bool, iptional
        Required to check for non-emptiness of the array. Default is
        'True'.

    Returns
    -------
    list
        A Python list containing the elements of `value`.

    Raises
    ------
    TypeError
        If `value` is not one of the supported list-like types.

    Notes
    -----
    NumPy arrays are converted using :meth:`numpy.ndarray.tolist`.
    """
    if isinstance(value, list):
        value = value
    elif isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    else:
        raise TypeError(f"{name} must be list-like (list/tuple/ndarray). Found {type(value)}.")
    if not allow_empty:
        if len(value) == 0:
            raise ValueError(f"{name} must not be empty.")
    return value


def _as_index_array(name: str, value: object) -> np.ndarray:
    """
    Convert an input object to a one-dimensional integer NumPy array.

    Parameters
    ----------
    name : str
        Name of the parameter being validated. Used only to produce
        informative error messages.
    value : object
        Input object to convert to a one-dimensional integer array.

    Returns
    -------
    np.ndarray
        A one-dimensional array of dtype `int`.

    Raises
    ------
    TypeError
        If `value` cannot be converted to a NumPy array of integers.
    ValueError
        If the resulting array is not one-dimensional or contains
        non-finite values.

    Notes
    -----
    This is a thin convenience wrapper around `_as_array` specialized
    for integer index vectors.
    """
    out = _as_array(name, value, dtype=int, ndim=1, strict_positive=True, allow_empty=False)
    return out


def _as_array(
    name: str,
    value: object,
    dtype=float,
    ndim: int | None = None,
    allow_empty: bool = True,
    non_negative: bool = False,
    strict_positive: bool = False,
) -> np.ndarray:
    """
    Convert an input object to a NumPy array and validate its shape
    and finiteness.

    Parameters
    ----------
    name : str
        Name of the parameter being validated. Used only to produce
        informative error messages.
    value : object
        Input object to convert to a NumPy array.
    dtype : data-type, optional
        Desired NumPy dtype of the output array. Default is `float`.
    ndim : int or None, optional
        Required number of dimensions. If provided, the converted array
        must have exactly this dimensionality. If `None`, dimensionality
        is not checked. Default is `None`.
    allow_empty: bool, iptional
        Required to check for non-emptiness of the array. Default is
        'True'.
    non_negative: bool, optional
        Required to check for non-negativity of the array. Default is
        'False'.
    strict_positive: bool, optional
        Required to check for strict positivity of the array. If active,
        silences the non_negative flag above. Default is 'False'.

    Returns
    -------
    numpy.ndarray
        The converted NumPy array.

    Raises
    ------
    TypeError
        If `value` cannot be converted to a NumPy array with the
        requested dtype.
    ValueError
        If `ndim` is specified and the resulting array does not have the
        required number of dimensions.
    ValueError
        If the resulting array contains non-finite values.
    ValueError
        If allow_empty is not enforced and the array is empty.
    ValueError
        If strict_positive is enforced and the array contains positive
        or zero values
    ValueError
        If non_negative is enforced and the array contains negatives
        values

    Notes
    -----
    Finiteness is checked with :func:`numpy.isfinite`, so arrays
    containing ``NaN`` or infinite values are rejected.
    """
    try:
        out = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} could not be converted to an ndarray.") from e

    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D. Found shape {out.shape}.")

    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values.")

    if not allow_empty and out.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if strict_positive:
        if not np.all(out > 0):
            raise ValueError(f"{name} must be strictly positive.")
    elif non_negative:
        if not np.all(out >= 0):
            raise ValueError(f"{name} must be non-negative.")

    return out


def _as_str(name: str, value: object) -> str:
    """
    Coerce a value to a string with strict semantics.

    Supports native strings and filesystem path objects
    (e.g. pathlib.Path) via the os.PathLike protocol.

    Parameters
    ----------
    name : str
        Name of the parameter being validated.
    value : object
        Value to convert.

    Returns
    -------
    str
        Converted string.

    Raises
    ------
    TypeError
        If the value cannot be converted to a string safely.
    ValueError
        If the resulting string is empty.
    """
    if isinstance(value, str):
        out = value

    elif isinstance(value, (bytes, bytearray)):
        try:
            out = value.decode("utf-8")
        except Exception as e:
            raise TypeError(f"{name} bytes could not be decoded as UTF-8.") from e

    else:
        try:
            # Handles pathlib.Path and any PathLike object
            out = os.fspath(value)
        except TypeError as e:
            raise TypeError(f"{name} must be str-like or PathLike. Found {type(value)}.") from e

    if len(out) == 0:
        raise ValueError(f"{name} must not be an empty string.")

    return out


def _as_file_path(name: str, value: object) -> Path:
    """
    Coerce a value to a filesystem path and validate that it is an existing file.

    Parameters
    ----------
    name : str
        Name of the parameter being validated.
    value : object
        Value to convert to a file path. Must be path-like.

    Returns
    -------
    Path
        Validated path object pointing to an existing file.

    Raises
    ------
    TypeError
        If the value cannot be interpreted as a path.
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path exists but is not a file.
    """
    try:
        path = Path(value).expanduser()
    except Exception as e:
        raise TypeError(f"{name} must be path-like (str or Path). Found {type(value)}.") from e

    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"{name} must be a file. Found directory: {path}")

    return path


def _as_dir_path(name: str, value: object) -> Path:
    """
    Coerce a value to a filesystem path and validate that it is an existing directory.

    Parameters
    ----------
    name : str
        Name of the parameter being validated.
    value : object
        Value to convert to a directory path. Must be path-like.

    Returns
    -------
    Path
        Validated path object pointing to an existing directory.

    Raises
    ------
    TypeError
        If the value cannot be interpreted as a path.
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path exists but is not a directory.
    """
    try:
        path = Path(value).expanduser()
    except Exception as e:
        raise TypeError(f"{name} must be path-like (str or Path). Found {type(value)}.") from e

    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    if not path.is_dir():
        raise ValueError(f"{name} must be a directory. Found file: {path}")

    return path
