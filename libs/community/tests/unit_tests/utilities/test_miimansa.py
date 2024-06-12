import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

from langchain_community.utilities.miimansa import MiimansaUtility


@pytest.fixture
def metadata():
    return {
        "qid2emb": {
            "1": np.array([[0.1, 0.2, 0.3]]),
            "2": np.array([[0.4, 0.5, 0.6]]),
        },
        "mean_vector": np.array([0.25, 0.35, 0.45]),
        "variance_vector": np.array([0.01, 0.01, 0.01]),
    }


@pytest.fixture
def direct_hit_model():
    model = MagicMock(spec=SentenceTransformer)
    model.encode.return_value = np.array([0.4, 0.5, 0.82])
    return model


def test_is_direct_hit(metadata, direct_hit_model):
    query = "test query"
    threshold = 0.8
    result = MiimansaUtility.is_direct_hit(query, metadata, direct_hit_model, threshold)
    assert result == (True, "2")
