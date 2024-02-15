import json
import pickle

import pandas as pd


def model_to_bytes(model):
    return pickle.dumps(model, protocol=pickle.DEFAULT_PROTOCOL)


def model_from_bytes(model_bytes):
    return pickle.loads(model_bytes)


def convert_json(_json):
    if pd.isnull(_json):
        return _json
    else:
        try:
            return json.loads(_json)
        except:  # noqa E722
            return None
