# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.LogSoftmax(dim=0).to(dev)
tensor = torch.ones(1,2,3,4).to(dev)
result = model(tensor)

mlir = torch_mlir.get_mlir( result )

# CHECK-LABEL: test_export_logsoftmax
#   CHECK: = "aten._log_softmax"(%arg0, %0, %1) {layer_name = "L0-_log_softmax-0"} : (tensor<1x2x3x4xf32>, i32, i1) -> tensor<1x2x3x4xf32>
print("test_export_logsoftmax")
print(mlir)

# CHECK-LABEL: test_jit_logsoftmax
#   CHECK: PASS!
print("test_jit_logsoftmax")
tensor = torch.randn(10).to(dev)
test_result = model(tensor)
ref_model = torch.nn.LogSoftmax(dim=0)

ref_result = ref_model(tensor.to('cpu'))
if test_result.to('cpu').equal( ref_result ):
    print ("PASS!")
else:
    print ("fail.")
