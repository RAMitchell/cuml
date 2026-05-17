# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import warnings
from collections.abc import Iterable

import cudf
import cupy as cp
import dask
import numpy as np
from dask.distributed import Future, get_worker
from raft_dask.common.comms import Comms, get_raft_comm_state

from cuml import using_output_type
from cuml.dask._compat import DASK_2025_4_0
from cuml.dask.common.input_utils import DistributedDataHandler, concatenate
from cuml.dask.common.utils import get_client, wait_and_raise_from_futures


class BaseRandomForestModel(object):
    """
    BaseRandomForestModel defines functions used in both Random Forest
    Classifier and Regressor for Multi Node and Multi GPU models. The common
    functions are defined here and called from the main Random Forest Multi
    Node Multi GPU APIs. The functions defined here are not meant to be used
    as a part of the public API.
    """

    def _create_model(
        self,
        model_func,
        client,
        workers,
        n_estimators,
        base_seed,
        ignore_empty_partitions,
        **kwargs,
    ):
        self.client = get_client(client)
        if workers is None:
            # Default to all workers
            client_kwargs = {"n_workers": -1} if DASK_2025_4_0() else {}
            workers = list(
                self.client.scheduler_info(**client_kwargs)["workers"].keys()
            )
        self.workers = workers
        self._set_internal_model(None)
        # rapids-pre-commit-hooks: disable-next-line
        # TODO(26.08): Drop along with the single-GPU deprecation in
        # cuml.ensemble.randomforest_common.
        # Record whether the user explicitly set `max_depth`; the warning is
        # emitted from `_fit` so it fires at fit time (matching the single-GPU
        # path) rather than at construction. We still forward an explicit
        # `max_depth=16` to the per-worker single-GPU estimators so they don't
        # each emit their own FutureWarning.
        self._max_depth_user_set = "max_depth" in kwargs
        if not self._max_depth_user_set:
            kwargs["max_depth"] = 16

        self.active_workers = list()
        self.ignore_empty_partitions = ignore_empty_partitions
        self.n_estimators = n_estimators

        if base_seed is None:
            base_seed = 0

        self.n_estimators_per_worker = [
            n_estimators for _ in range(len(self.workers))
        ]

        self.rfs = {
            worker: self.client.submit(
                model_func,
                n_estimators=n_estimators,
                random_state=base_seed,
                **kwargs,
                pure=False,
                workers=[worker],
            )
            for worker in self.workers
        }

        wait_and_raise_from_futures(list(self.rfs.values()))

    def _estimators_per_worker(self, n_estimators):
        return [n_estimators for _ in range(len(self.workers))]

    def _fit(self, model, dataset, convert_dtype, broadcast_data):
        # rapids-pre-commit-hooks: disable-next-line
        # TODO(26.08): Drop along with the single-GPU deprecation in
        # cuml.ensemble.randomforest_common.
        if not getattr(self, "_max_depth_user_set", True):
            warnings.warn(
                "The default value of 'max_depth' will change from 16 to "
                # rapids-pre-commit-hooks: disable-next-line
                "None (unlimited depth) in release 26.08. To suppress this "
                "warning, set 'max_depth' explicitly.",
                FutureWarning,
                stacklevel=3,
            )

        if broadcast_data:
            warnings.warn(
                "broadcast_data is ignored for distributed RandomForest "
                "training because each worker participates in one global "
                "tree build over its local rows.",
                UserWarning,
                stacklevel=3,
            )

        data = DistributedDataHandler.create(dataset, client=self.client)
        fit_workers = list(self.workers)
        self.active_workers = fit_workers
        self.datatype = data.datatype

        labels = self.client.persist(dataset[1])
        if self.datatype == "cudf":
            self.num_classes = len(labels.unique())
        else:
            self.num_classes = len(dask.array.unique(labels).compute())

        global_n_rows = sum(total for _, total in data._worker_sizes.values())
        if global_n_rows <= 0:
            raise ValueError("RandomForest requires at least one global row")

        n_cols = dataset[0].shape[1]
        x_dtype = _dtype_from_input(dataset[0])
        y_dtype = _dtype_from_input(dataset[1])
        classes = getattr(self, "unique_classes", None)

        comms = Comms(
            comms_p2p=False, client=self.client, streams_per_handle=1
        )
        comms.init(workers=fit_workers)
        futures = []
        fit_futures = {}
        try:
            for worker in fit_workers:
                worker_data = data.worker_to_parts.get(worker)
                fit_future = self.client.submit(
                    _func_fit_distributed,
                    model[worker],
                    comms.sessionId,
                    worker_data,
                    convert_dtype,
                    self.datatype,
                    x_dtype,
                    y_dtype,
                    n_cols,
                    global_n_rows,
                    classes,
                    workers=[worker],
                    pure=False,
                )
                fit_futures[worker] = fit_future
                futures.append(fit_future)
            wait_and_raise_from_futures(futures)
        finally:
            comms.destroy()

        self.rfs.update(fit_futures)
        self.n_active_estimators_per_worker = [
            self.n_estimators for _ in self.active_workers
        ]
        self._set_internal_model(futures[0].result())
        return self

    def _concat_treelite_models(self):
        """
        Return one worker model.

        Distributed training now synchronizes the core tree builder across
        workers, so each worker owns an equivalent full forest. The old Dask
        implementation concatenated independent per-worker sub-forests here.
        """
        return self.rfs[self.active_workers[0]].result()

    def _partial_inference(self, X, op_type, delayed, **kwargs):
        data = DistributedDataHandler.create(X, client=self.client)
        combined_data = list(map(lambda x: x[1], data.gpu_futures))

        if op_type == "classification":
            func = _func_predict_proba_partial
            shape = (X.shape[0], 1, self.num_classes)
        else:
            shape = (X.shape[0], 1)
            func = _func_predict_partial

        meta = cp.zeros((0,) * len(shape), dtype=cp.float32)

        partial_infs = list()
        for worker in self.active_workers:
            partial_infs.append(
                self.client.submit(
                    func,
                    self.rfs[worker],
                    combined_data,
                    **kwargs,
                    workers=[worker],
                    pure=False,
                )
            )

        objs = [
            dask.array.from_delayed(partial_inf, shape=shape, meta=meta)
            for partial_inf in partial_infs
        ]
        result = dask.array.concatenate(objs, axis=1)
        return result

    def _predict_using_fil(self, X, delayed, **kwargs):
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        data = DistributedDataHandler.create(X, client=self.client)
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        return self._predict(
            X, delayed=delayed, output_collection_type=data.datatype, **kwargs
        )

    def _get_params(self, deep):
        model_params = list()
        for idx, worker in enumerate(self.workers):
            model_params.append(
                self.client.submit(
                    _func_get_params, self.rfs[worker], deep, workers=[worker]
                )
            )
        params_of_each_model = self.client.gather(model_params, errors="raise")
        return params_of_each_model

    def _set_params(self, **params):
        # rapids-pre-commit-hooks: disable-next-line
        # TODO(26.08): Drop along with the single-GPU deprecation in
        # cuml.ensemble.randomforest_common.
        if "max_depth" in params:
            self._max_depth_user_set = True
        model_params = list()
        for idx, worker in enumerate(self.workers):
            model_params.append(
                self.client.submit(
                    _func_set_params,
                    self.rfs[worker],
                    **params,
                    workers=[worker],
                )
            )
        wait_and_raise_from_futures(model_params)
        return self

    def get_combined_model(self):
        """
        Return single-GPU model for serialization.

        Returns
        -------

        model : Trained single-GPU model or None if the model has not
                yet been trained.
        """

        # set internal model if it hasn't been accessed before
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())

        internal_model = self._check_internal_model(self._get_internal_model())

        if isinstance(self.internal_model, Iterable):
            # This function needs to return a single instance of cuml.Base,
            # even if the class is just a composite.
            raise ValueError(
                "Expected a single instance of cuml.Base "
                "but got %s instead." % type(self.internal_model)
            )

        elif isinstance(self.internal_model, Future):
            internal_model = self.internal_model.result()

        return internal_model

    def _get_workers_weights(self) -> cp.ndarray:
        workers_weights = np.array(self.n_active_estimators_per_worker)
        workers_weights = workers_weights[workers_weights != 0]
        workers_weights = workers_weights / workers_weights.sum()
        workers_weights = cp.array(workers_weights)
        return workers_weights

    def apply_reduction(self, reduce, partial_infs, datatype, delayed):
        """
        Reduces the partial inferences to obtain the final result. The workers
        didn't have the same number of trees to form their predictions. To
        correct for this worker's predictions are weighted differently during
        reduction.
        """
        workers_weights = self._get_workers_weights()
        unique_classes = (
            None
            if not hasattr(self, "unique_classes")
            else self.unique_classes
        )
        delayed_local_array = dask.delayed(reduce)(
            partial_infs, workers_weights, unique_classes
        )
        delayed_res = dask.array.from_delayed(
            delayed_local_array, shape=(np.nan, np.nan), dtype=np.float32
        )
        if delayed:
            return delayed_res
        else:
            return delayed_res.persist()


def _func_fit(model, input_data, convert_dtype):
    X = concatenate([item[0] for item in input_data])
    y = concatenate([item[1] for item in input_data])
    return model.fit(X, y, convert_dtype=convert_dtype)


def _func_fit_distributed(
    model,
    session_id,
    input_data,
    convert_dtype,
    datatype,
    x_dtype,
    y_dtype,
    n_cols,
    global_n_rows,
    classes,
):
    state = get_raft_comm_state(session_id, get_worker())
    handle = state["handle"]
    if input_data is None:
        X, y = _empty_local_data(datatype, x_dtype, y_dtype, n_cols)
    else:
        X = concatenate([item[0] for item in input_data])
        y = concatenate([item[1] for item in input_data])
    if hasattr(model, "_fit_with_handle"):
        kwargs = {
            "convert_dtype": convert_dtype,
            "global_n_rows": global_n_rows,
        }
        if classes is not None:
            kwargs["classes"] = classes
        return model._fit_with_handle(X, y, handle, **kwargs)
    return model.fit(X, y, convert_dtype=convert_dtype)


def _empty_local_data(datatype, x_dtype, y_dtype, n_cols):
    if datatype == "cudf":
        X = cudf.DataFrame(
            {i: cp.empty(0, dtype=x_dtype) for i in range(n_cols)}
        )
        y = cudf.Series(cp.empty(0, dtype=y_dtype))
    else:
        X = cp.empty((0, n_cols), dtype=x_dtype, order="F")
        y = cp.empty(0, dtype=y_dtype)
    return X, y


def _dtype_from_input(data):
    dtype = getattr(data, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)

    meta = getattr(data, "_meta", None)
    dtype = getattr(meta, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)

    dtypes = getattr(meta, "dtypes", None)
    if dtypes is not None:
        if hasattr(dtypes, "iloc"):
            return np.dtype(dtypes.iloc[0])
        return np.dtype(next(iter(dtypes)))

    raise TypeError(f"Could not determine dtype for {type(data)!r}")


def _func_predict_partial(model, input_data, **kwargs):
    """
    Whole dataset inference with part of the model (trees at disposal locally).
    Transfer dataset instead of model. Interesting when model is larger
    than dataset.
    """
    X = concatenate(input_data)
    with using_output_type("cupy"):
        prediction = model.predict(X, **kwargs)
        return cp.expand_dims(prediction, axis=1)


def _func_predict_proba_partial(model, input_data, **kwargs):
    """
    Whole dataset inference with part of the model (trees at disposal locally).
    Transfer dataset instead of model. Interesting when model is larger
    than dataset.
    """
    X = concatenate(input_data)
    with using_output_type("cupy"):
        prediction = model.predict_proba(X, **kwargs)
        return cp.expand_dims(prediction, axis=1)


def _func_get_params(model, deep):
    return model.get_params(deep)


def _func_set_params(model, **params):
    return model.set_params(**params)


def _serialize_treelite_bytes(model):
    return model._treelite_model_bytes
