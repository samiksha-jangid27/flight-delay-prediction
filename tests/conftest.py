"""Pytest session setup.

macOS OpenMP conflict guard
---------------------------
PyTorch and XGBoost each ship their own OpenMP runtime: ``torch`` loads its
bundled ``torch/lib/libomp.dylib`` while ``xgboost`` loads the Homebrew
``/opt/homebrew/opt/libomp`` copy. With both active in one process (the test
suite imports ``torch`` for the DL tests and deserializes a saved XGBoost
pipeline in ``test_xgboost.py``), macOS segfaults inside whichever OpenMP
runtime loaded second.

Setting these two environment variables *before* any test module imports
``torch`` or ``xgboost`` lets the two runtimes coexist for the duration of the
test session:

* ``KMP_DUPLICATE_LIB_OK=TRUE`` — tolerate a second OpenMP runtime in-process.
* ``OMP_NUM_THREADS=1`` — single-threaded OpenMP, which avoids the crash and
  keeps test timing deterministic.

This file runs at pytest startup, before test collection, so the variables are
in place before the libraries initialize. It affects the test process only;
training and evaluation scripts run in their own processes and keep full
multithreading. ``setdefault`` means an explicit value in the environment wins.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
