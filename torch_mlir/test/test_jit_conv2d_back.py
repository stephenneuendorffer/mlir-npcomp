# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

N = 3
Cin = 16
Cout = 4
w = 10
h = 10

model = torch.nn.Conv2d(Cin, Cout, (3,3))
ref_model = torch.nn.Conv2d(Cin, Cout, (3,3))

ref_model.weight.data = model.weight.clone()
ref_model.bias.data = model.bias.clone()

model = model.to(dev)

softmax = torch.nn.LogSoftmax(dim=1)
loss = torch.nn.NLLLoss()

tensor = torch.randn(N, Cin, h, w, device=dev)
result = model(tensor)

# CHECK-LABEL: test_jit_conv2d
#   CHECK: PASS!
print("test_jit_conv2d")
ref_result = ref_model(tensor.to('cpu'))

error = (ref_result - result.to('cpu')).abs().max()
if error <= 1e-5:
    print ("PASS!")
else:
    print ("fail.")

print(torch_mlir.get_mlir( result ))

target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, Cout)
ref_target = target.clone()
target = target.to(dev)

test_loss = loss( softmax(result), target )
test_loss.backward()

ref_loss = loss( softmax(ref_result), ref_target )
ref_loss.backward()

print(torch_mlir.get_mlir( test_loss ))

# CHECK-LABEL: test_jit_conv2d_loss
#   CHECK: PASS!
print("test_jit_conv2d_loss")
err = (ref_loss - test_loss.to('cpu')).abs().max()
print(err)
if err <= 1e-5:
    print ("PASS!")
else:
    print ("fail.")

print(torch_mlir.get_mlir( model.weight.grad ))

# CHECK-LABEL: test_jit_conv2d_weight_grad
#   CHECK: PASS!
print("test_jit_conv2d_weight_grad")
err = (ref_model.weight.grad - model.weight.grad.to('cpu')).abs().max()
if err <= 1e-5:
    print ("PASS!")
else:
    print ("fail.")

# CHECK-LABEL: test_jit_conv2d_bias_grad
#   CHECK: PASS!
print("test_jit_conv2d_bias_grad")
err = (ref_model.bias.grad - model.bias.grad.to('cpu')).abs().max()
if err <= 1e-5:
    print ("PASS!")
else:
    print ("fail.")
