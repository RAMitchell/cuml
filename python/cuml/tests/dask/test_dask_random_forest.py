# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cupy as cp
import dask_cudf
import numpy as np
import pandas as pd
import pytest
import treelite
from dask.array import from_array
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from cuml.dask.common import utils as dask_utils
from cuml.dask.ensemble import RandomForestClassifier as cuRFC_mg
from cuml.dask.ensemble import RandomForestRegressor as cuRFR_mg
from cuml.ensemble import RandomForestClassifier as cuRFC_sg
from cuml.ensemble import RandomForestRegressor as cuRFR_sg

# rapids-pre-commit-hooks: disable-next-line
# TODO(26.08): Remove this filter
pytestmark = pytest.mark.filterwarnings(
    "ignore:The default value of 'max_depth':FutureWarning"
)


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, y_train_df = dask_utils.persist_across_workers(
        c, [X_train_df, y_train_df], workers=workers
    )
    return X_train_df, y_train_df


def _assert_num_trees(model, n_estimators):
    treelite_bytes = model.internal_model._treelite_model_bytes
    treelite_model = treelite.Model.deserialize_bytes(treelite_bytes)
    assert treelite_model.num_tree == n_estimators


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_regression_dask_fil(partitions_per_worker, dtype, client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    X, y = make_regression(
        n_samples=n_workers * 4000,
        n_features=20,
        n_informative=10,
        random_state=123,
    )

    X = X.astype(dtype)
    y = y.astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 100, random_state=123
    )

    cu_rf_params = {
        "n_estimators": 50,
        "max_depth": 16,
        "n_bins": 16,
    }

    workers = client.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)

    X_cudf = cudf.DataFrame(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)
    X_cudf_test = cudf.DataFrame(pd.DataFrame(X_test))
    X_test_df = dask_cudf.from_cudf(X_cudf_test, npartitions=n_partitions)

    cuml_mod = cuRFR_mg(**cu_rf_params, ignore_empty_partitions=True)
    cuml_mod.fit(X_train_df, y_train_df)
    _assert_num_trees(cuml_mod, cu_rf_params["n_estimators"])

    cuml_mod_predict = cuml_mod.predict(X_test_df)
    cuml_mod_predict = cp.asnumpy(cp.array(cuml_mod_predict.compute()))

    acc_score = r2_score(y_test, cuml_mod_predict)

    assert acc_score >= 0.59


@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_classification_dask_array(partitions_per_worker, client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    X, y = make_classification(
        n_samples=n_workers * 2000,
        n_features=30,
        n_clusters_per_class=1,
        n_informative=20,
        random_state=123,
        n_classes=2,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 400
    )

    cu_rf_params = {
        "n_estimators": 25,
        "max_depth": 13,
        "n_bins": 15,
    }

    X_train_df, y_train_df = _prep_training_data(
        client, X_train, y_train, partitions_per_worker
    )
    X_test_dask_array = from_array(X_test)
    cuml_mod = cuRFC_mg(**cu_rf_params)
    cuml_mod.fit(X_train_df, y_train_df)
    _assert_num_trees(cuml_mod, cu_rf_params["n_estimators"])
    cuml_mod_predict = cuml_mod.predict(X_test_dask_array).compute()

    acc_score = accuracy_score(cuml_mod_predict, y_test, normalize=True)

    assert acc_score > 0.8


@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_classification_dask_fil_predict_proba(
    partitions_per_worker, client
):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    X, y = make_classification(
        n_samples=n_workers * 1500,
        n_features=30,
        n_clusters_per_class=1,
        n_informative=20,
        random_state=123,
        n_classes=2,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 150, random_state=123
    )

    cu_rf_params = {
        "n_bins": 16,
        "n_streams": 1,
        "n_estimators": 40,
        "max_depth": 16,
    }

    X_train_df, y_train_df = _prep_training_data(
        client, X_train, y_train, partitions_per_worker
    )
    X_test_df, _ = _prep_training_data(
        client, X_test, y_test, partitions_per_worker
    )
    cu_rf_mg = cuRFC_mg(**cu_rf_params)
    cu_rf_mg.fit(X_train_df, y_train_df)

    fil_preds = cu_rf_mg.predict(X_test_df).compute()
    fil_preds = fil_preds.to_numpy()
    fil_preds_proba = cu_rf_mg.predict_proba(X_test_df).compute()
    fil_preds_proba = fil_preds_proba.to_numpy()
    np.testing.assert_equal(fil_preds, np.argmax(fil_preds_proba, axis=1))

    y_proba = np.zeros(np.shape(fil_preds_proba))
    y_proba[:, 1] = y_test
    y_proba[:, 0] = 1.0 - y_test
    fil_mse = mean_squared_error(y_proba, fil_preds_proba)
    sk_model = skrfc(
        n_estimators=cu_rf_params["n_estimators"],
        max_depth=cu_rf_params["max_depth"],
        random_state=10,
    )
    sk_model.fit(X_train, y_train)
    sk_preds_proba = sk_model.predict_proba(X_test)
    sk_mse = mean_squared_error(y_proba, sk_preds_proba)

    # The threshold is required as the test would intermitently
    # fail with a max difference of 0.029 between the two mse values
    assert fil_mse <= sk_mse + 0.029


@pytest.mark.parametrize("estimator_type", ["regression", "classification"])
def test_rf_get_combined_model_right_aftter_fit(client, estimator_type):
    max_depth = 3
    n_estimators = 5

    X, y = make_classification()
    X = X.astype(np.float32)
    if estimator_type == "classification":
        cu_rf_mg = cuRFC_mg(
            max_features=1.0,
            max_samples=1.0,
            n_bins=16,
            n_streams=1,
            n_estimators=n_estimators,
            max_leaves=-1,
            max_depth=max_depth,
        )
        y = y.astype(np.int32)
    elif estimator_type == "regression":
        cu_rf_mg = cuRFR_mg(
            max_features=1.0,
            max_samples=1.0,
            n_bins=16,
            n_streams=1,
            n_estimators=n_estimators,
            max_leaves=-1,
            max_depth=max_depth,
        )
        y = y.astype(np.float32)
    else:
        assert False
    X_dask, y_dask = _prep_training_data(client, X, y, partitions_per_worker=2)
    cu_rf_mg.fit(X_dask, y_dask)
    single_gpu_model = cu_rf_mg.get_combined_model()
    if estimator_type == "classification":
        assert isinstance(single_gpu_model, cuRFC_sg)
    elif estimator_type == "regression":
        assert isinstance(single_gpu_model, cuRFR_sg)
    else:
        assert False
