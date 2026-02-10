sph_lib
=======

SPH particle-to-grid deposition and interpolation utilities.

Quick start
-----------

.. code-block:: python

   import numpy as np
   from sph_lib import PointCloud

   positions = np.random.rand(1000, 2).astype(np.float32)
   masses = np.random.rand(1000).astype(np.float32)

   sim = PointCloud(positions=positions, masses=masses, boxsize=1.0, verbose=False)
   fields = masses[:, None]

   grid = sim.deposit_to_grid(
       fields=fields,
       averaged=[False],
       gridnums=64,
       method="cic",
       use_python=False,
   )

API reference
-------------

.. toctree::
   :maxdepth: 2

   api
   collection
   kernels
   grid
   utils
