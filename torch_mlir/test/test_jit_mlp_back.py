# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torch_mlir

# RUN: python %s | FileCheck %s

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

def main():

    #device = 'cpu'
    device = torch_mlir.mlir_device()

    model = Net()
    ref_model = Net()

    ref_model.fc1.weight.data = model.fc1.weight.clone()
    ref_model.fc1.bias.data = model.fc1.bias.clone()
    ref_model.fc2.weight.data = model.fc2.weight.clone()
    ref_model.fc2.bias.data = model.fc2.bias.clone()
    ref_model.fc3.weight.data = model.fc3.weight.clone()
    ref_model.fc3.bias.data = model.fc3.bias.clone()

    model = model.to(device)

    ref_tensor = torch.randn((64, 1, 28, 28),requires_grad=True)
    tensor = ref_tensor.clone().to(device)

    tensor.retain_grad()
    ref_tensor.retain_grad()

    result = model(tensor)
    ref_result = ref_model(ref_tensor)

    # CHECK-LABEL: test_export_mlp_fwd
    #   CHECK: return %{{.*}} : tensor<64x10xf32>
    print("test_export_mlp_fwd")
    mlir = torch_mlir.get_mlir(result)
    print(mlir)

    # CHECK-LABEL: test_jit_mlp_fwd
    #   CHECK: PASS! fwd result check
    print("test_jit_mlp_fwd")
    err = (ref_result - result.to('cpu')).abs().max()
    if (err <= 1e-5):
        print ("PASS! fwd result check")
    else:
        print ("failed fwd result check")

    target = torch.ones((64), dtype=torch.long, device=device)
    loss = F.nll_loss(result, target)
    loss.backward()

    ref_target = torch.ones((64), dtype=torch.long)
    ref_loss = F.nll_loss(ref_result, ref_target)
    ref_loss.backward()

    # CHECK-LABEL: test_jit_mlp_loss
    #   CHECK: PASS! fwd loss check
    print("test_jit_mlp_loss")
    err = (ref_loss - loss.to('cpu')).abs().max()
    if (err <= 1e-5):
        print ("PASS! fwd loss check")
    else:
        print ("failed fwd loss check")

    # CHECK-LABEL: test_export_mlp_back
    #   CHECK: return %{{.*}} : tensor<50x784xf32>
    print("test_export_mlp_back")
    mlir = torch_mlir.get_mlir(model.fc1.weight.grad)
    print(mlir)

    # CHECK-LABEL: test_jit_mlp_fc1_weight_grad
    #   CHECK: PASS! back grad check
    print("test_jit_mlp_fc1_weight_grad")
    err = (model.fc1.weight.grad.to('cpu') - ref_model.fc1.weight.grad).abs().max()
    if (err <= 1e-5):
        print ("PASS! back grad check")
    else:
        print ("failed back grad check")
    #print (torch_mlir.get_mlir(model.fc1.weight.grad))
    #print("max error", (model.fc1.weight.grad.to('cpu') - ref_model.fc1.weight.grad).abs().max())

    #print (tensor.grad.to('cpu')*2)
    #print (ref_tensor.grad)


if __name__ == '__main__':
    main()
