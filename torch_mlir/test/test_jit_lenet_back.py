# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_mlir
#torch_mlir.set_debug(True)

# RUN: python %s | FileCheck %s

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
        #self.maxpool2d = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(9216*4, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #x = self.maxpool2d(x)
        x = x.view((64,9216*4))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def main():

    device = torch_mlir.mlir_device()

    model = Net()
    ref_model = Net()

    ref_model.conv1.weight.data = model.conv1.weight.clone()
    ref_model.conv1.bias.data = model.conv1.bias.clone()
    ref_model.conv2.weight.data = model.conv2.weight.clone()
    ref_model.conv2.bias.data = model.conv2.bias.clone()
    ref_model.fc1.weight.data = model.fc1.weight.clone()
    ref_model.fc1.bias.data = model.fc1.bias.clone()
    ref_model.fc2.weight.data = model.fc2.weight.clone()
    ref_model.fc2.bias.data = model.fc2.bias.clone()

    model = model.to(device)

    def back_hook(self, input, output):
        if (self == model.conv2):
            print ("TEST")
        else:
            print ("REF")
        if not isinstance(output, tuple):
            output = (output,)
        if not isinstance(input, tuple):
            input = (input,)
        nin = 0
        nout = 0
        for i in input:
            i = i.to('cpu')
            print ("NIN", nin, i)
            nin = nin+1
        for o in output:
            o = o.to('cpu')
            print ("NOUT", nout, o)
            nout = nout+1

    #model.conv2.register_backward_hook(back_hook)
    #ref_model.conv2.register_backward_hook(back_hook)

    ref_tensor = torch.randn((64, 1, 28, 28), requires_grad=True)
    tensor = ref_tensor.clone().to(device)

    tensor.retain_grad()
    ref_tensor.retain_grad()

    result = model(tensor)
    ref_result = ref_model(ref_tensor)

    target = torch.ones((64), dtype=torch.long).to(device)
    loss = F.nll_loss(result, target)
    loss.backward()

    ref_target = torch.ones((64), dtype=torch.long)
    ref_loss = F.nll_loss(ref_result, ref_target)
    ref_loss.backward()

    # CHECK-LABEL: test_export_lenet_back
    #   CHECK: return {{.*}} : tensor<64x32x3x3xf32>
    print("test_export_lenet_back")
    mlir = torch_mlir.get_mlir(model.conv2.weight.grad)
    print (mlir)

    # CHECK-LABEL: fwd_loss
    #   CHECK: PASS! fwd loss check
    print ("fwd_loss")
    with torch.no_grad():
        loss_err = (ref_loss - loss.to('cpu')).abs().max()
    if loss_err < 1e-5:
        print ("PASS! fwd loss check")
    else:
        print ("failed fwd loss check")

    # CHECK-LABEL: conv2_weight_grad_check
    #   CHECK: PASS! conv2 weight grad check
    print ("conv2_weight_grad_check")
    with torch.no_grad():
        cpu = model.conv2.weight.grad.to('cpu')
        err = (ref_model.conv2.weight.grad - cpu).abs().max()
    if err < 1e-5:
        print ("PASS! conv2 weight grad check")
    else:
        print ("failed conv2 weight grad check")

    # CHECK-LABEL: conv1_bias_grad_check
    #   CHECK: PASS! conv1 bias grad check
    print ("conv1_bias_grad_check")
    with torch.no_grad():
        cpu = model.conv1.bias.grad.to('cpu')
        err = (ref_model.conv1.bias.grad - cpu).abs().max()
    if err < 1e-5:
        print ("PASS! conv1 bias grad check")
    else:
        print ("failed conv1 bias grad check")

    # CHECK-LABEL: fc1_weight_grad_check
    #   CHECK: PASS! fc1 weight grad check
    print ("fc1_weight_grad_check")
    with torch.no_grad():
        cpu = model.fc1.weight.grad.to('cpu')
        err = (ref_model.fc1.weight.grad - cpu).abs().max()
    if err < 1e-5:
        print ("PASS! fc1 weight grad check")
    else:
        print ("failed fc1 weight grad check")

if __name__ == '__main__':
    main()
