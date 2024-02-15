from typing import Callable

import pandas as pd


def apply_to_partition(func: Callable, *args, **kwargs):
    def do_work(partition: pd.DataFrame):
        return partition.apply(func, axis=1, args=args, **kwargs)

    return do_work
