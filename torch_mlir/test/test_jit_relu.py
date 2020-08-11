# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.ReLU().to(dev)
tensor = torch.ones(1,2,3,4).to(dev)
result = model(tensor)

mlir = torch_mlir.get_mlir( result )

# CHECK-LABEL: test_export_relu
#   CHECK: = "aten.relu"(%{{.*}}) {layer_name = "L0-relu-0"} : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
print("test_export_relu")
print(mlir)

# CHECK-LABEL: test_jit_relu
#   CHECK: PASS!
print("test_jit_relu")
tensor = torch.randn(10).to(dev)
test_result = model(tensor)
ref_model = torch.nn.ReLU()
ref_result = ref_model(tensor.to('cpu'))
if test_result.to('cpu').equal( ref_result ):
    print ("PASS!")
else:
    print ("fail.")
