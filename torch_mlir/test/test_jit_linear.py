# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.Linear(1024,16).to(dev)
tensor = torch.randn(4,1024).to(dev)
result = model(tensor)

mlir = torch_mlir.get_mlir( result )

# CHECK-LABEL: test_export_linear
#   CHECK: [[V0:%[a-zA-Z0-9]+]] = "aten.t"(%{{.*}}) {layer_name = "L0-t-0"} : (tensor<16x1024xf32>) -> tensor<1024x16xf32>
#   CHECK: [[V1:%[a-zA-Z0-9]+]] = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
#   CHECK: [[V2:%[a-zA-Z0-9]+]] = "aten.addmm"(%{{.*}}, %{{.*}}, [[V0]], [[V1]], [[V1]]) {layer_name = "L1-addmm-0"} : (tensor<16xf32>, tensor<4x1024xf32>, tensor<1024x16xf32>, i32, i32) -> tensor<4x16xf32>
#   CHECK: return [[V2]] : tensor<4x16xf32>
print("test_export_linear")
print(mlir)

# CHECK-LABEL: test_jit_linear
#   CHECK: PASS!
print("test_jit_linear")

ref_model = model.to('cpu')
ref_result = ref_model(tensor.to('cpu'))
result = result.to('cpu')

error = (ref_result - result).abs().max()
if error <= 1e-5:
    print ("PASS!")
else:
    print ("fail.")
