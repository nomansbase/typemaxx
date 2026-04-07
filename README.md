# typemaxx

Lightweight typing aliases and validation utilities for NumPy-based workflows.

## Overview

`typemaxx` provides two main components:

- Structured type aliases built on top of `jaxtyping`
- Explicit validation helpers for coercion and input checking

It is designed for:
- scientific / numerical Python code
- predictable runtime validation
- clearer function signatures

Dependencies:
- numpy
- jaxtyping

---

## Installation

### Local (development)

```bash
pip install -e .
```

### From Git

```bash
pip install "typemaxx @ git+ssh://git@github.com/you/typemaxx.git"
```

---

## Typing utilities

Example:

```python
from typemaxx.types import HWCArray, MaskArray
```

Available aliases:

- Array — generic float array
- HWCArray — (H, W, C)
- CHWArray — (C, H, W)
- HWArray — (H, W)
- PixelsArray — (L, C)
- ImageArray — (X, Y, Z)
- MaskArray — integer (H, W)
- LineArray — (L,)
- IndexArray — integer (N,)

Also:

```python
JSONValue
```

Recursive JSON-compatible type.

---

## Validation utilities

All validation functions follow the same structure:

- convert input
- validate constraints
- raise explicit exceptions

---

### Scalars

```python
_as_float(name, value, non_negative=False, strict_positive=False)
_as_int(name, value, non_negative=False, strict_positive=False)
```

Features:
- type coercion
- finiteness checks (float)
- optional sign constraints

---

### Arrays

```python
_as_array(name, value, dtype=float, ndim=None, allow_empty=True, non_negative=False, strict_positive=False)
_as_index_array(name, value)
```

Features:
- conversion via `numpy.asarray`
- dimensionality checks
- finiteness checks
- optional non-empty enforcement
- optional sign constraints

---

### Containers

```python
_as_list(name, value, allow_empty=True)
```

Supports:
- list
- tuple
- numpy.ndarray

---

### Strings and paths

```python
_as_str(name, value)
_as_file_path(name, value)
_as_dir_path(name, value)
```

Features:
- strict string coercion
- UTF-8 decoding for bytes
- PathLike support
- filesystem validation

---

## Design principles

- explicit over implicit
- small, composable utilities
- deterministic error handling
- minimal dependencies

---

## Example

```python
from typemaxx.validation import _as_array
from typemaxx.types import HWCArray

def normalize_image(x: HWCArray) -> HWCArray:
    x = _as_array("x", x, dtype=float, ndim=3, non_negative=True)
    return x / x.max()
```

---

## Project structure

```
src/
  typemaxx/
    types.py
    validation.py
```

---

## License

Apache-2.0 License.