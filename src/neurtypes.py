from typing import Annotated, TypeAlias, Literal

import numpy as np

Case: TypeAlias = np.ndarray[tuple[int], np.dtype[np.unsignedinteger]]

State: TypeAlias = Annotated[np.ndarray[tuple[int, int], np.dtype[np.unsignedinteger]],   tuple[Case]]

Diff: TypeAlias = np.ndarray[tuple[int, Literal[2], int], np.dtype[np.unsignedinteger]]

Emit: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
