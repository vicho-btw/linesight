"""
Optional-numba shim.

numba ships unsigned native extensions (`_dynfunc.pyd`, llvmlite's `.dll`), and Windows
Smart App Control / WDAC blocks loading them:

    ImportError: DLL load failed while importing _dynfunc:
    An Application Control policy has blocked this file.

Rather than require the machine's code-integrity policy be weakened, fall back to running
the decorated functions as plain Python. Everything numba is used for in this project is a
small numeric helper -- a few vector norms, an np.interp, one short while loop -- so the
functions stay correct and the cost is minor next to rollout and training time.

Import `jit` / `njit` from here instead of from numba.
"""

try:
    from numba import jit, njit  # noqa: F401

    NUMBA_AVAILABLE = True
except Exception as _numba_import_error:  # pragma: no cover - depends on host policy
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = _numba_import_error

    def jit(*args, **kwargs):
        """No-op stand-in supporting both @jit and @jit(nopython=True) spellings."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    njit = jit
