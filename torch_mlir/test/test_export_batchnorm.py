# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.BatchNorm2d(123).to(dev)
result = model(torch.ones(42,123,4,5).to(dev))

# CHECK-LABEL: test_export_batchnorm
#       CHECK: %0 = "aten.constant"() {type = "bool", value = true} : () -> i1
#       CHECK: %1 = "aten.constant"() {type = "f32", value = 1.000000e-01 : f32} : () -> f32
#       CHECK: %2 = "aten.constant"() {type = "f32", value = 9.99999974E-6 : f32} : () -> f32
#       CHECK: %output, %save_mean, %save_invstd = "aten.native_batch_norm"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %0, %1, %2) {layer_name = "L0-native_batch_norm-0"} : (tensor<42x123x4x5xf32>, tensor<123xf32>, tensor<123xf32>, tensor<123xf32>, tensor<123xf32>, i1, f32, f32) -> (tensor<42x123x4x5xf32>, tensor<123xf32>, tensor<123xf32>)
mlir = torch_mlir.get_mlir( result )
print("test_export_batchnorm")
print(mlir)