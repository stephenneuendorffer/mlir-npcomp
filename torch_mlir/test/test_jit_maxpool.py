# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1),
                           dilation=1, return_indices=False, ceil_mode=False).to(dev)

tensor = torch.randn(1,32,16,16).to(dev)
result = model(tensor)

# CHECK-LABEL: test_export_maxpool
#       CHECK: %0 = "aten.constant"() {type = "List[i32]", value = dense<3> : vector<2xi32>} : () -> !aten.list<i32>
#       CHECK: %1 = "aten.constant"() {type = "List[i32]", value = dense<2> : vector<2xi32>} : () -> !aten.list<i32>
#       CHECK: %2 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
#       CHECK: %3 = "aten.constant"() {type = "bool", value = false} : () -> i1
#       CHECK: %4:2 = "aten.max_pool2d_with_indices"(%arg0, %0, %1, %2, %2, %3) {layer_name = "L0-max_pool2d_with_indices-0"} : (tensor<1x32x16x16xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>, i1) -> (tensor<1x32x8x8xf32>, tensor<1x32x8x8xi64>)
mlir = torch_mlir.get_mlir( result )
print("test_export_maxpool")
print(mlir)

# CHECK-LABEL: test_jit_maxpool
#   CHECK: PASS!
print("test_jit_maxpool")
ref_model = model.to('cpu')
ref_result = ref_model(tensor.to('cpu'))
result = result.to('cpu')

error = (ref_result - result).abs().max()
if error <= 1e-5:
    print ("PASS!")
else:
    print ("fail.")
