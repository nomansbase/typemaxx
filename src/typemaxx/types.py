from __future__ import annotations

from typing import TypeAlias

import numpy as np
from jaxtyping import Float, Int

JSONValue: TypeAlias = (
    dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None
)

Array: TypeAlias = Float[np.ndarray, "*"]
HWCArray: TypeAlias = Float[np.ndarray, "H W C"]
CHWArray: TypeAlias = Float[np.ndarray, "C H W"]
HWArray: TypeAlias = Float[np.ndarray, "H W"]
PixelsArray: TypeAlias = Float[np.ndarray, "L C"]
ImageArray: TypeAlias = Float[np.ndarray, "X Y Z"]
MaskArray: TypeAlias = Int[np.ndarray, "H W"]
LineArray: TypeAlias = Float[np.ndarray, "L"]
IndexArray: TypeAlias = Int[np.ndarray, "N"]
