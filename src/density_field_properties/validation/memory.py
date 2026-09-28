"""Memory helpers for validation pipeline stages."""

import gc


def release_validation_memory() -> None:
    """
    Close matplotlib figures and run garbage collection.

    Call this after each validation stage so large arrays and figures from a
    previous step are not retained while the next one runs.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    gc.collect()
