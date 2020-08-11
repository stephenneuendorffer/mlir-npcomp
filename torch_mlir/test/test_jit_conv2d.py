# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.Conv2d(2,16,7,stride=[2,2], padding=[3,3], dilation=1, groups=1, bias=True).to(dev)

tensor = torch.randn((1,2,128,128), device=dev)
result = model(tensor)

mlir = torch_mlir.get_mlir( result )
# CHECK-LABEL: test_export_conv2d
#   CHECK: %0 = "aten.constant"() {type = "List[i32]", value = dense<2> : vector<2xi32>} : () -> !aten.list<i32>
#   CHECK: %1 = "aten.constant"() {type = "List[i32]", value = dense<3> : vector<2xi32>} : () -> !aten.list<i32>
#   CHECK: %2 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
#   CHECK: %3 = "aten.constant"() {type = "bool", value = false} : () -> i1
#   CHECK: %4 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi32>} : () -> !aten.list<i32>
#   CHECK: %5 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
#   CHECK: = "aten.convolution_overrideable"(%arg0, %arg1, %arg2, %0, %1, %2, %3, %4, %5) {layer_name = "L0-convolution_overrideable-0"} : (tensor<1x2x128x128xf32>, tensor<16x2x7x7xf32>, tensor<16xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>, i1, !aten.list<i32>, i32) -> tensor<1x16x64x64xf32>
print("test_export_conv2d")
print( mlir )

# CHECK-LABEL: test_jit_conv2d
#   CHECK: PASS!
print("test_jit_conv2d")
ref_model = model.to('cpu')
ref_result = ref_model(tensor.to('cpu'))
result = result.to('cpu')

error = (ref_result - result).abs().max()
if error <= 1e-5:
    print ("PASS!")
else:
    print ("fail.")
