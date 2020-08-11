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
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.maxpool2d = nn.MaxPool2d(2,2)
        #self.dropout1 = nn.Dropout2d(0.25)
        #self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxpool2d(x)
        #x = self.dropout1(x)
        x = x.view((4,9216))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main():

    device = torch_mlir.mlir_device()

    model = Net().to(device)
    ref_tensor = torch.randn((4, 1, 28, 28))
    tensor = ref_tensor.clone().to(device)

    result = model(tensor)

    ref_model = model.to('cpu')
    ref_result = ref_model( ref_tensor)

    # CHECK-LABEL: test_jit_lenet_fwd
    #   CHECK: PASS!
    print("test_jit_lenet_fwd")
    err = (ref_result - result.to('cpu')).abs().max()
    if (err <= 1e-5):
        print ("PASS!")
    else:
        print ("fail.")

if __name__ == '__main__':
    main()
