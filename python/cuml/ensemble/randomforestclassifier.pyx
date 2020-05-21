
#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import ctypes
import cudf
import cupy as cp
import math
import numpy as np
import rmm
import warnings

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cython.operator cimport dereference as deref

from cuml import ForestInference
from cuml.fil.fil import TreeliteModel

from cuml.common.array import CumlArray
from cuml.common.handle import Handle
from cuml.ensemble.randomforest_common import BaseRandomForestModel

from cuml.common.handle cimport cumlHandle
from cuml.ensemble.randomforest_common import _obtain_treelite_model, \
    _obtain_fil_model
from cuml.ensemble.randomforest_shared cimport *
import cuml.common.logger as logger
from cuml.common import input_to_cuml_array, rmm_cupy_ary

from numba import cuda

cimport cuml.common.handle
cimport cuml.common.cuda

cimport cython


cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":

    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[float, int]*,
                  float*,
                  int,
                  int,
                  int*,
                  int,
                  RF_params,
                  int) except +

    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[double, int]*,
                  double*,
                  int,
                  int,
                  int*,
                  int,
                  RF_params,
                  int) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[float, int] *,
                      float*,
                      int,
                      int,
                      int*,
                      bool) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[double, int]*,
                      double*,
                      int,
                      int,
                      int*,
                      bool) except +

    cdef void predictGetAll(cumlHandle& handle,
                            RandomForestMetaData[float, int] *,
                            float*,
                            int,
                            int,
                            int*,
                            bool) except +

    cdef void predictGetAll(cumlHandle& handle,
                            RandomForestMetaData[double, int]*,
                            double*,
                            int,
                            int,
                            int*,
                            bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[float, int]*,
                          int*,
                          int,
                          int*,
                          bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[double, int]*,
                          int*,
                          int,
                          int*,
                          bool) except +


class RandomForestClassifier(BaseRandomForestModel):
    """
    Implements a Random Forest classifier model which fits multiple decision
    tree classifiers in an ensemble.

    Note that the underlying algorithm for tree node splits differs from that
    used in scikit-learn. By default, the cuML Random Forest uses a
    histogram-based algorithms to determine splits, rather than an exact
    count. You can tune the size of the histograms with the n_bins parameter.

    **Known Limitations**: This is an early release of the cuML
    Random Forest code. It contains a few known limitations:

       * GPU-based inference is only supported if the model was trained
         with 32-bit (float32) datatypes. CPU-based inference may be used
         in this case as a slower fallback.
       * Very deep / very wide models may exhaust available GPU memory.
         Future versions of cuML will provide an alternative algorithm to
         reduce memory consumption.

    Examples
    ---------
    .. code-block:: python

            import numpy as np
            from cuml.ensemble import RandomForestClassifier as cuRFC

            X = np.random.normal(size=(10,4)).astype(np.float32)
            y = np.asarray([0,1]*5, dtype=np.int32)

            cuml_model = cuRFC(max_features=1.0,
                               n_bins=8,
                               n_estimators=40)
            cuml_model.fit(X,y)
            cuml_predict = cuml_model.predict(X)

            print("Predicted labels : ", cuml_predict)

    Output:

    .. code-block:: none

            Predicted labels :  [0 1 0 1 0 1 0 1 0 1]

    Parameters
    -----------
    n_estimators : int (default = 100)
        Number of trees in the forest. (Default changed to 100 in cuML 0.11)
    handle : cuml.Handle
        If it is None, a new one is created just for this class.
    split_criterion : The criterion used to split nodes.
        0 for GINI, 1 for ENTROPY
        2 and 3 not valid for classification
        (default = 0)
    split_algo : int (default = 1)
        The algorithm to determine how nodes are split in the tree.
        0 for HIST and 1 for GLOBAL_QUANTILE. HIST curently uses a slower
        tree-building algorithm so GLOBAL_QUANTILE is recommended for most
        cases.
    bootstrap : boolean (default = True)
        Control bootstrapping.
        If True, each tree in the forest is built
        on a bootstrapped sample with replacement.
        If False, sampling without replacement is done.
    bootstrap_features : boolean (default = False)
        Control bootstrapping for features.
        If features are drawn with or without replacement
    rows_sample : float (default = 1.0)
        Ratio of dataset rows used while fitting each tree.
    max_depth : int (default = 16)
        Maximum tree depth. Unlimited (i.e, until leaves are pure),
        if -1. Unlimited depth is not supported.
        *Note that this default differs from scikit-learn's
        random forest, which defaults to unlimited depth.*
    max_leaves : int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited,
        if -1.
    max_features : int, float, or string (default = 'auto')
        Ratio of number of features (columns) to consider per node split.
        If int then max_features/n_features.
        If float then max_features is used as a fraction.
        If 'auto' then max_features=1/sqrt(n_features).
        If 'sqrt' then max_features=1/sqrt(n_features).
        If 'log2' then max_features=log2(n_features)/n_features.
    n_bins : int (default = 8)
        Number of bins used by the split algorithm.
    min_rows_per_node : int or float (default = 2)
        The minimum number of samples (rows) needed to split a node.
        If int then number of sample rows.
        If float the min_rows_per_sample*n_rows
    min_impurity_decrease : float (default = 0.0)
        Minimum decrease in impurity requried for
        node to be spilt.
    quantile_per_tree : boolean (default = False)
        Whether quantile is computed for individal trees in RF.
        Only relevant for GLOBAL_QUANTILE split_algo.
    seed : int (default = None)
        Seed for the random number generator. Unseeded by default.
    """
    def __init__(self, split_criterion=0, seed=None,
                 n_streams=8, **kwargs):

        if ((seed is not None) and (n_streams != 1)):
            warnings.warn("For reproducible results, n_streams==1 is "
                          "recommended. If n_streams is > 1, results may vary "
                          "due to stream/thread timing differences, even when "
                          "random_seed is set")

        self.RF_type = CLASSIFICATION
        self.num_classes = 2
        self._create_model(model=RandomForestClassifier,
                           split_criterion=split_criterion,
                           seed=seed, n_streams=n_streams,
                           **kwargs)

    """
    TODO:
        Add the preprocess and postprocess functions
        in the cython code to normalize the labels
        Link to the above issue on github :
        https://github.com/rapidsai/cuml/issues/691
    """
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['handle']
        cdef size_t params_t
        cdef  RandomForestMetaData[float, int] *rf_forest
        cdef  RandomForestMetaData[double, int] *rf_forest64
        cdef size_t params_t64
        if self.n_cols:
            # only if model has been fit previously
            self._get_protobuf_bytes()  # Ensure we have this cached
            if self.rf_forest:
                params_t = <uintptr_t> self.rf_forest
                rf_forest = \
                    <RandomForestMetaData[float, int]*>params_t
                state["rf_params"] = rf_forest.rf_params

            if self.rf_forest64:
                params_t64 = <uintptr_t> self.rf_forest64
                rf_forest64 = \
                    <RandomForestMetaData[double, int]*>params_t64
                state["rf_params64"] = rf_forest64.rf_params

        state['n_cols'] = self.n_cols
        state["verbosity"] = self.verbosity
        state["model_pbuf_bytes"] = self.model_pbuf_bytes
        state["treelite_handle"] = None

        return state

    def __setstate__(self, state):
        super(RandomForestClassifier, self).__init__(
            handle=None, verbosity=state['verbosity'])
        cdef  RandomForestMetaData[float, int] *rf_forest = \
            new RandomForestMetaData[float, int]()
        cdef  RandomForestMetaData[double, int] *rf_forest64 = \
            new RandomForestMetaData[double, int]()

        self.n_cols = state['n_cols']
        if self.n_cols:
            rf_forest.rf_params = state["rf_params"]
            state["rf_forest"] = <uintptr_t>rf_forest

            rf_forest64.rf_params = state["rf_params64"]
            state["rf_forest64"] = <uintptr_t>rf_forest64

        self.model_pbuf_bytes = state["model_pbuf_bytes"]
        self.__dict__.update(state)

    def __del__(self):
        self._reset_forest_data()

    def _reset_forest_data(self):
        """Free memory allocated by this instance and clear instance vars."""
        if self.rf_forest:
            delete_rf_metadata(
                <RandomForestMetaData[float, int]*><uintptr_t>
                self.rf_forest)
            self.rf_forest = 0
        if self.rf_forest64:
            delete_rf_metadata(
                <RandomForestMetaData[double, int]*><uintptr_t>
                self.rf_forest64)
            self.rf_forest64 = 0

        if self.treelite_handle:
            TreeliteModel.free_treelite_model(self.treelite_handle)

        self.treelite_handle = None
        self.model_pbuf_bytes = bytearray()
        self.n_cols = None


    def _obtain_treelite_handle(self):
        """Returns a handle to a treelite-formatted version of the model.
        This will create a new treelite model if necessary, or return
        a cached version when available. The handle is cached in the
        instanced and freed at instance deletion. Caller should not
        delete the returned model."""
        if self.treelite_handle is not None:
            return self.treelite_handle  # Cached version

        cdef ModelHandle cuml_model_ptr = NULL
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        assert len(self.model_pbuf_bytes) > 0 or self.rf_forest, \
            "Attempting to create treelite from un-fit forest."

        if self.num_classes > 2:
            raise NotImplementedError("Pickling for multi-class "
                                      "classification models is currently not "
                                      "implemented. Please check cuml issue "
                                      "#1679 for more information.")
        cdef unsigned char[::1] model_pbuf_mv = self.model_pbuf_bytes
        cdef vector[unsigned char] model_pbuf_vec
        with cython.boundscheck(False):
            model_pbuf_vec.assign(& model_pbuf_mv[0],
                                  & model_pbuf_mv[model_pbuf_mv.shape[0]])

        task_category = CLASSIFICATION_MODEL
        build_treelite_forest(
            & cuml_model_ptr,
            rf_forest,
            <int> self.n_cols,
            <int> self.num_classes,
            model_pbuf_vec)
        mod_ptr = <uintptr_t> cuml_model_ptr
        self.treelite_handle = ctypes.c_void_p(mod_ptr).value
        return self.treelite_handle

    def convert_to_treelite_model(self):
        """
        Converts the cuML RF model to a Treelite model

        Returns
        ----------
        tl_to_fil_model : Treelite version of this model
        """
        handle = self._obtain_treelite_handle()
        return _obtain_treelite_model(handle)

    def convert_to_fil_model(self, output_class=True,
                             threshold=0.5, algo='auto',
                             fil_sparse_format='auto'):
        """
        Create a Forest Inference (FIL) model from the trained cuML
        Random Forest model.

        Parameters
        ----------
        output_class : boolean (default = True)
            This is optional and required only while performing the
            predict operation on the GPU.
            If true, return a 1 or 0 depending on whether the raw
            prediction exceeds the threshold. If False, just return
            the raw prediction.
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
            multiple rows per thread block
            `auto` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU.
            It is applied if output_class == True, else it is ignored
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.
            'auto' - choose the storage type automatically
            (currently True is chosen by auto)
            False - create a dense forest
            True - create a sparse forest, requires algo='naive'
            or algo='auto'

        Returns
        ----------
        fil_model :
            A Forest Inference model which can be used to perform
            inferencing on the random forest model.
        """
        treelite_handle = self._obtain_treelite_handle()
        return _obtain_fil_model(treelite_handle=treelite_handle,
                                 depth=self.max_depth,
                                 output_class=output_class,
                                 threshold=threshold,
                                 algo=algo,
                                 fil_sparse_format=fil_sparse_format)

    """
    TODO : Move functions duplicated in the RF classifier and regressor
           to a shared file. Cuml issue #1854 has been created to track this.
    """

    def fit(self, X, y, convert_dtype=False):
        """
        Perform Random Forest Classification on the input data

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        y : array-like (device or host) shape = (n_samples, 1)
            Dense vector (int32) of shape (n_samples, 1).
            Acceptable formats: NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
            These labels should be contiguous integers from 0 to n_classes.
        convert_dtype : bool, optional (default = False)
            When set to True, the fit method will, when necessary, convert
            y to be the same data type as X if they differ. This will increase
            memory used for the method.

        """
        cdef uintptr_t X_ptr, y_ptr
        X_m, y_m, max_feature_val = self._dataset_setup(X, y, convert_dtype)
        X_ptr = X_m.ptr
        y_ptr = y_m.ptr
        cdef cumlHandle* handle_ =\
            <cumlHandle*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            new RandomForestMetaData[float, int]()
        self.rf_forest = <uintptr_t> rf_forest
        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            new RandomForestMetaData[double, int]()
        self.rf_forest64 = <uintptr_t> rf_forest64

        if self.seed is None:
            seed_val = <uintptr_t>NULL
        else:
            seed_val = <uintptr_t>self.seed

        rf_params = set_rf_class_obj(<int> self.max_depth,
                                     <int> self.max_leaves,
                                     <float> max_feature_val,
                                     <int> self.n_bins,
                                     <int> self.split_algo,
                                     <int> self.min_rows_per_node,
                                     <float> self.min_impurity_decrease,
                                     <bool> self.bootstrap_features,
                                     <bool> self.bootstrap,
                                     <int> self.n_estimators,
                                     <float> self.rows_sample,
                                     <int> seed_val,
                                     <CRITERION> self.split_criterion,
                                     <bool> self.quantile_per_tree,
                                     <int> self.n_streams)

        if self.dtype == np.float32:
            fit(handle_[0],
                rf_forest,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> self.num_classes,
                rf_params,
                <int> self.verbosity)

        elif self.dtype == np.float64:
            rf_params64 = rf_params
            fit(handle_[0],
                rf_forest64,
                <double*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> self.num_classes,
                rf_params64,
                <int> self.verbosity)

        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))
        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del X_m
        del y_m
        return self

    def _predict_model_on_cpu(self, X, convert_dtype):
        out_type = self._get_output_type(X)
        cdef uintptr_t X_ptr
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C',
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        X_ptr = X_m.ptr

        preds = CumlArray.zeros(n_rows, dtype=np.int32)
        cdef uintptr_t preds_ptr = preds.ptr

        cdef cumlHandle* handle_ =\
            <cumlHandle*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64
        if self.dtype == np.float32:
            predict(handle_[0],
                    rf_forest,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <int> self.verbosity)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    rf_forest64,
                    <double*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <int> self.verbosity)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        # synchronous w/o a stream
        del(X_m)
        return preds.to_output(out_type)

    def predict(self, X, predict_model="GPU",
                output_class=True, threshold=0.5,
                algo='auto',
                convert_dtype=True,
                fil_sparse_format='auto'):
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        predict_model : String (default = 'GPU')
            'GPU' to predict using the GPU, 'CPU' otherwise. The 'GPU' can only
            be used if the model was trained on float32 data and `X` is float32
            or convert_dtype is set to True. Also the 'GPU' should only be
            used for binary classification problems.
        output_class : boolean (default = True)
            This is optional and required only while performing the
            predict operation on the GPU.
            If true, return a 1 or 0 depending on whether the raw
            prediction exceeds the threshold. If False, just return
            the raw prediction.
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
            multiple rows per thread block
            `auto` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU.
            It is applied if output_class == True, else it is ignored
        num_classes : int (default = 2)
            number of different classes present in the dataset
        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.
            'auto' - choose the storage type automatically
            (currently True is chosen by auto)
            False - create a dense forest
            True - create a sparse forest, requires algo='naive'
            or algo='auto'

        Returns
        ----------
        y : NumPy
           Dense vector (int) of shape (n_samples, 1)
        """
        if predict_model == "CPU" or self.num_classes > 2:
            if self.num_classes > 2 and predict_model == "GPU":
                warnings.warn("Switching over to use the CPU predict since "
                              "the GPU predict currently cannot perform "
                              "multi-class classification.")
            preds = self._predict_model_on_cpu(X, convert_dtype)

        elif self.dtype == np.float64:
            raise TypeError("GPU based predict only accepts np.float32 data. \
                            In order use the GPU predict the model should \
                            also be trained using a np.float32 dataset. \
                            If you would like to use np.float64 dtype \
                            then please use the CPU based predict by \
                            setting predict_model = 'CPU'")

        else:
            preds = \
                self._predict_model_on_gpu(X=X, output_class=output_class,
                                           threshold=threshold,
                                           algo=algo,
                                           convert_dtype=convert_dtype,
                                           fil_sparse_format=fil_sparse_format,
                                           predict_proba=False)

        return preds

    def _predict_get_all(self, X, convert_dtype=True):
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        Returns
        ----------
        y : NumPy
           Dense vector (int) of shape (n_samples, 1)
        """
        out_type = self._get_output_type(X)
        cdef uintptr_t X_ptr, preds_ptr
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C',
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        X_ptr = X_m.ptr

        preds = CumlArray.zeros(n_rows * self.n_estimators, dtype=np.int32)
        preds_ptr = preds.ptr

        cdef cumlHandle* handle_ =\
            <cumlHandle*><uintptr_t>self.handle.getHandle()
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64
        if self.dtype == np.float32:
            predictGetAll(handle_[0],
                          rf_forest,
                          <float*> X_ptr,
                          <int> n_rows,
                          <int> n_cols,
                          <int*> preds_ptr,
                          <int> self.verbosity)

        elif self.dtype == np.float64:
            predictGetAll(handle_[0],
                          rf_forest64,
                          <double*> X_ptr,
                          <int> n_rows,
                          <int> n_cols,
                          <int*> preds_ptr,
                          <int> self.verbosity)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))
        self.handle.sync()
        del(X_m)
        return preds.to_output(out_type)

    def predict_proba(self, X, output_class=True,
                      threshold=0.5, algo='auto',
                      convert_dtype=True,
                      fil_sparse_format='auto'):
        """
        Predicts class probabilites for X. This function uses the GPU
        implementation of predict. Therefore, data with 'dtype = np.float32'
        and 'num_classes = 2' should be used while using this function.
        The option to use predict_proba for multi_class classification is not
        currently implemented. Please check cuml issue #1679 for more
        information.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        output_class: boolean (default = True)
            This is optional and required only while performing the
            predict operation on the GPU.
            If true, return a 1 or 0 depending on whether the raw
            prediction exceeds the threshold. If False, just return
            the raw prediction.
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
            multiple rows per thread block
            `auto` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU.
            It is applied if output_class == True, else it is ignored
        num_classes : int (default = 2)
            number of different classes present in the dataset
        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.
            'auto' - choose the storage type automatically
            (currently True is chosen by auto)
            False - create a dense forest
            True - create a sparse forest, requires algo='naive'
            or algo='auto'

        Returns
        -------
        y : (same as the input datatype)
            Dense vector (float) of shape (n_samples, 1). The datatype of y
            depend on the value of 'output_type' varaible specified by the
            user while intializing the model.
        """
        if self.dtype == np.float64:
            raise TypeError("GPU based predict only accepts np.float32 data. \
                            In order use the GPU predict the model should \
                            also be trained using a np.float32 dataset. \
                            If you would like to use np.float64 dtype \
                            then please use the CPU based predict by \
                            setting predict_model = 'CPU'")

        elif self.num_classes > 2:
            raise NotImplementedError("Predict_proba for multi-class "
                                      "classification models is currently not "
                                      "implemented. Please check cuml issue "
                                      "#1679 for more information.")
        preds_proba = \
            self._predict_model_on_gpu(X, output_class=output_class,
                                       threshold=threshold,
                                       algo=algo,
                                       convert_dtype=convert_dtype,
                                       fil_sparse_format=fil_sparse_format,
                                       predict_proba=True)

        return preds_proba

    def score(self, X, y, threshold=0.5,
              algo='auto', num_classes=2, predict_model="GPU",
              convert_dtype=True, fil_sparse_format='auto'):
        """
        Calculates the accuracy metric score of the model for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        y : NumPy
            Dense vector (int) of shape (n_samples, 1)
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
            multiple rows per thread block
            `auto` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        threshold : float
            threshold is used to for classification
            This is optional and required only while performing the
            predict operation on the GPU.
        num_classes : integer
            number of different classes present in the dataset
        convert_dtype : boolean, default=True
            whether to convert input data to correct dtype automatically
        predict_model : String (default = 'GPU')
            'GPU' to predict using the GPU, 'CPU' otherwise. The 'GPU' can only
            be used if the model was trained on float32 data and `X` is float32
            or convert_dtype is set to True. Also the 'GPU' should only be
            used for binary classification problems.
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.
            'auto' - choose the storage type automatically
            (currently True is chosen by auto)
            False - create a dense forest
            True - create a sparse forest, requires algo='naive'
            or algo='auto'

        Returns
        -------
        accuracy : float
           Accuracy of the model [0.0 - 1.0]
        """
        cdef uintptr_t X_ptr, y_ptr
        _, n_rows, _, _ = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        y_m, n_rows, _, y_dtype = \
            input_to_cuml_array(y, check_dtype=np.int32,
                                convert_to_dtype=(np.int32 if convert_dtype
                                                  else False))
        y_ptr = y_m.ptr
        preds = self.predict(X, output_class=True,
                             threshold=threshold, algo=algo,
                             convert_dtype=convert_dtype,
                             predict_model=predict_model,
                             fil_sparse_format=fil_sparse_format)

        cdef uintptr_t preds_ptr
        preds_m, _, _, _ = \
            input_to_cuml_array(preds, convert_to_dtype=np.int32)
        preds_ptr = preds_m.ptr

        cdef cumlHandle* handle_ =\
            <cumlHandle*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float32:
            self.stats = score(handle_[0],
                               rf_forest,
                               <int*> y_ptr,
                               <int> n_rows,
                               <int*> preds_ptr,
                               <int> self.verbosity)
        elif self.dtype == np.float64:
            self.stats = score(handle_[0],
                               rf_forest64,
                               <int*> y_ptr,
                               <int> n_rows,
                               <int*> preds_ptr,
                               <int> self.verbosity)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        del(y_m)
        del(preds_m)
        return self.stats['accuracy']

    def get_params(self, deep=True):
        """
        Returns the value of all parameters
        required to configure this estimator as a dictionary.

        Parameters
        -----------
        deep : boolean (default = True)
        """


        return self._get_params(model=RandomForestClassifier,
                                deep=deep)

    def set_params(self, **params):
        """
        Sets the value of parameters required to
        configure this estimator, it functions similar to
        the sklearn set_params.

        Parameters
        -----------
        params : dict of new params
        """
        # Resetting handle as __setstate__ overwrites with handle=None


        return self._set_params(model=RandomForestClassifier,
                                **params)

    def print_summary(self):
        """
        Prints the summary of the forest used to train and test the model
        """
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            print_rf_summary(rf_forest64)
        else:
            print_rf_summary(rf_forest)

    def print_detailed(self):
        """
        Prints the detailed information about the forest used to
        train and test the Random Forest model
        """
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            print_rf_detailed(rf_forest64)
        else:
            print_rf_detailed(rf_forest)
