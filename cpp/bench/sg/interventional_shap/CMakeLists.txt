# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
cmake_minimum_required(VERSION 3.30)
project(interventional_shap_experiment LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(raft REQUIRED)
find_package(rmm REQUIRED)
find_package(Treelite REQUIRED)
include(FetchContent)
FetchContent_Declare(
  gputreeshap
  GIT_REPOSITORY https://github.com/rapidsai/gputreeshap.git
  GIT_TAG 93292317b23ef733f881c881865f5d5728dc2fea
)
FetchContent_MakeAvailable(gputreeshap)
add_library(interventional_shap_experiment SHARED driver.cu)
set_target_properties(
  interventional_shap_experiment
  PROPERTIES CXX_STANDARD 20
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_include_directories(interventional_shap_experiment PRIVATE ../../../include)
target_link_libraries(
  interventional_shap_experiment PRIVATE raft::raft rmm::rmm treelite::treelite
                                         GPUTreeShap::GPUTreeShap CUDA::cudart
)
target_compile_options(
  interventional_shap_experiment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                                         --expt-relaxed-constexpr>
)
