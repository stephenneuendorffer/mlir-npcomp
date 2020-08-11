# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.LogSoftmax(dim=1).to(dev)
loss = torch.nn.NLLLoss().to(dev)

input = torch.randn(3,5,requires_grad=True,device=dev)

target = torch.tensor([1, 0, 4])
result = loss(model(input), target)

mlir = torch_mlir.get_mlir( result )

# CHECK-LABEL: test_export_nllloss
#   CHECK: :2 = "aten.nll_loss_forward"(%2, %arg1, %arg2, %0, %3) {layer_name = "L1-nll_loss_forward-0"} : (tensor<3x5xf32>, tensor<3xi64>, tensor<5xf32>, i32, i32) -> (tensor<1xf32>, tensor<1xf32>)
print("test_export_nllloss")
print(mlir)

# CHECK-LABEL: test_jit_nllloss
#   CHECK: PASS!
print("test_jit_nllloss")

ref_model = torch.nn.LogSoftmax(dim=1)
ref_loss = torch.nn.NLLLoss()

ref_result = ref_loss(ref_model(input.to('cpu')), target.to('cpu'))
err = (ref_result - result.to('cpu')).abs().max()
if (err <= 1e-5):
    print ("PASS!")
else:
    print ("fail.")
