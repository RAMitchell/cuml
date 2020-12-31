#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cuml
import cupy as cp
import numpy as np

from cuml.common.input_utils import input_to_cupy_array
from cuml.experimental.explainer.base import SHAPBase
from cuml.experimental.explainer.common import get_cai_ptr
from cuml.experimental.explainer.common import get_dtype_from_model_func
from cuml.experimental.explainer.common import get_tag_from_model_func
from cuml.experimental.explainer.common import model_func_call
from cuml.experimental.explainer.common import output_list_shap_values


class ImportanceExplainer(SHAPBase):
    def __init__(self,
                 model,
                 masker,
                 masker_type='independent',
                 link='identity',
                 handle=None,
                 is_gpu_model=None,
                 random_state=None,
                 dtype=None,
                 output_type=None,
                 verbose=False, ):
        super(ImportanceExplainer, self).__init__(
            order='C',
            model=model,
            background=masker,
            link=link,
            verbose=verbose,
            random_state=random_state,
            is_gpu_model=is_gpu_model,
            handle=handle,
            dtype=dtype,
            output_type=output_type
        )

        self._synth_data = None

    def shap_values(self,
                    X, max_evals=100
                    ,
                    main_effects=False):
        return self._explain(X,
                             max_evals=max_evals,
                             main_effects=main_effects)

    def _explain(self,
                 X,
                 max_evals=None,
                 main_effects=False
                 ):

        X = input_to_cupy_array(X, order=self.order,
                                convert_to_dtype=self.dtype)[0]

        if X.ndim == 1:
            X = X.reshape((1, self.M))

        shap_values = []
        for i in range(self.D):
            shap_values.append(cp.zeros(X.shape, dtype=self.dtype))

        # Allocate synthetic dataset array once for multiple explanations
        if self._synth_data is None:
            self._synth_data = cp.zeros(
                shape=(max_evals * self.N, self.M),
                dtype=self.dtype,
                order=self.order
            )

        for idx, x in enumerate(X):
            # use mutability of lists and cupy arrays to get all shap values
            self._explain_single_observation(
                shap_values[idx],
                x.reshape(1, self.M),
                main_effects=main_effects,
                max_evals=max_evals,
                idx=idx
            )

        return output_list_shap_values(shap_values, self.D, self.output_type)

    def _explain_single_observation(self,
                                    shap_values,
                                    row,
                                    main_effects,
                                    max_evals,
                                    idx,
                                    ):

        # Draw samples
        mask = cp.random.binomial(1, 0.5, (max_evals, self.M))

        mask_kernel = """
        size_t col = i % ncols;
        size_t background_row =(i/ncols) % nbackground_rows;
        size_t eval_idx = i/(ncols*nbackground_rows);
        out = mask[eval_idx*ncols + col] ? foreground[col] : background[background_row*ncols + col];
        """
        mask_foreground = cp.ElementwiseKernel(
            'raw X background, raw Y foreground, raw Z mask, int64 ncols, int64 nbackground_rows',
            'W out', mask_kernel, 'mask_foreground')

        mask_foreground(self.background, row, mask, self.M, self.N, self._synth_data)

        # evaluate model on combinations
        y = model_func_call(X=self._synth_data,
                            model_func=self.model,
                            gpu_model=self.is_gpu_model)

        # accumulate shap values

        accumulate_preamble = """
inline __host__ __device__ double W(double s, double n) {
  return exp(n*log(2.0)+lgamma(s + 1) - lgamma(n + 1) + lgamma(n - s));
}
        """

        accumulate_kernel = """
            size_t eval_idx = i/(ncols*nbackground_rows);
            size_t col= i%ncols;
            size_t s_len=s_length[eval_idx];
            D f_eval= function_evals[i/ncols];
            if(mask[eval_idx*ncols + col]){
                A x = f_eval * W(s_len - 1, ncols) / (nbackground_rows*nevals);
                atomicAdd(&shap_values[col], x);
            }
            else{
                A x = - f_eval * W(s_len, ncols) / (nbackground_rows*nevals);
                atomicAdd(&shap_values[col], x);
            }
            
        """

        s_length = mask.sum(axis=1)

        accumulate_shap = cp.ElementwiseKernel(
            'raw B mask, raw C s_length, raw D function_evals, int64 ncols, '
            'int64 nbackground_rows, int64 nevals',
            'raw A shap_values', accumulate_kernel, preamble=accumulate_preamble)

        accumulate_shap(mask, s_length, y, self.M, self.N,max_evals, shap_values,size=max_evals*self.M*self.N)
