# +
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import scipy.spatial as spatial

from .python import functions as pyfunc
from .cpp import functions as cppfunc


def p2g(
	positions: np.ndarray,
	quantities: np.ndarray,
	averaged: Sequence[bool],
	gridnum: int,
	boxsizes: np.ndarray,
	periodic: Union[int, Sequence[bool]],
	*,
	method: Optional[str] = None,
	hsm: Optional[np.ndarray] = None,
	hmat_eigvecs: Optional[np.ndarray] = None,
	hmat_eigvals: Optional[np.ndarray] = None,
	return_weights: bool = False,
	use_python: bool = False,
	kernel_name: str = 'quintic',
	integration: str = 'midpoint',
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
	"""Dispatch particle-to-grid deposition to the selected backend.

	Parameters
	----------
	positions
		Particle coordinates with shape (N, dim).
	quantities
		Fields to deposit with shape (N, num_fields).
	averaged
		Flags indicating which fields should be normalized by weights.
	gridnum
		Number of cells per dimension.
	boxsizes
		Domain extents for each axis expressed as maximum lengths.
	periodic
		Per-axis flags.
	method
		Name of the deposition method (e.g., "ngp", "cic", "tsc").

	Returns
	-------
	Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
		Deposited fields, optionally accompanied by weights.
	"""

	if method is None:
		raise ValueError("p2g requires 'method' to be specified")
	
	dim = positions.shape[-1]
	boxsizes = np.asarray(boxsizes)

	# type casting
	positions  = positions.astype('float32')
	quantities = quantities.astype('float32')
	boxsizes   = boxsizes.astype('float32')

	# TODO: this does not work due to boxsizes, need to move this one 
	# to the ops.py file and compute the pcellsizehalfs there
	# we can also rename the pcellsizeshalf to smoothing lengths directly
	# and adapt the adaptive cic function accordingly
	

	# select the deposition function based on the method, dim and use_python
	namespace = pyfunc if use_python else cppfunc
	func = getattr(namespace, f"{method}_{dim}d")
	# print me the function name here for debugging
	print(f"Using deposition function: {func.__name__}")

	# perform deposition
	if "adaptive" in method:
		args = (
			positions,
			quantities,
			boxsizes,
			gridnum,
			periodic,
			hsm,
		)

	elif "isotropic" == method:
		args = (
			positions,
			quantities,
			boxsizes,
			gridnum,
			periodic,
			hsm,
			kernel_name,
			integration,
		)

	elif "anisotropic" == method:
		args = (
			positions,
			quantities,
			boxsizes,
			gridnum,
			periodic,
			hmat_eigvecs,
			hmat_eigvals,
			kernel_name,
			integration,
		)

	else:
		args = (
			positions,
			quantities,
			boxsizes,
			gridnum,
			periodic,
		)

	fields, weights = func(*args)

	# divide averaged fields by weight
	for i in range(len(averaged)):
		if averaged[i] == True:
			fields[..., i] /= (weights + 1e-10)

	if return_weights:
		return fields, weights
	else:
		return fields

