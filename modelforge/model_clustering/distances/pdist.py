import numpy as np
from scipy.spatial.distance import pdist

from modelforge.shared.logger import logger_factory

logger = logger_factory(__name__)


def pdist_wrapper(x, metric, *, out=None, **kwargs):
    if type(x) is np.ndarray:
        return pdist(x, metric, out=out)
    else:
        return pdist_obj(x, metric, out=out, **kwargs)


def pdist_obj(
    x, metric, client, dtype=np.float64, out=None, batch_size=10_000, **kwargs
):
    """
    Compute the distance between each pair of the input collection. This is an adaption of scipy.spatial.distance.pdist
    to work with objects instead of arrays.
    @param x: The input collection
    @param metric: The distance metric to use
    @param client: The client object to use
    @param dtype: Data type of the returned matrix
    @param kwargs: Additional arguments to pass to the distance metric
    @return: The distance metric in vector form
    """
    n = len(x)
    out_size = (n * (n - 1)) // 2
    expected_shape = (out_size,)
    if out is None:
        dm = np.empty(expected_shape, dtype=dtype)
    else:
        if out.shape != expected_shape:
            raise ValueError("Output array has incorrect shape.")
        if out.dtype != dtype:
            raise ValueError("Output array has incorrect type.")
        dm = out

    # Create a list to hold the delayed computations
    delayed_results = []
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if k % batch_size == 0:
                logger.info(f"Distance Metric Progress {k}/{out_size}")
                # Compute all the results in parallel
                intermediate_results = client.gather(delayed_results)
                delayed_results = []
                # Assign the results to the results array
                base_index = k - batch_size
                for m, result in enumerate(intermediate_results):
                    dm[base_index + m] = result
            # Wrap the distance calculation with client.submit
            result = client.submit(metric, x[i], x[j], **kwargs)
            delayed_results.append(result)
            k += 1

    # Gather any remaining results
    if delayed_results:
        intermediate_results = client.gather(delayed_results)
        base_index = k - len(delayed_results)
        for m, result in enumerate(intermediate_results):
            dm[base_index + m] = result

    return dm
