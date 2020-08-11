# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_mlir

import inspect

# RUN: python %s | FileCheck %s

class ResA(nn.Module):
    def __init__(self, channels):
      C = int(channels)
      C2 = int(channels/2)
      super(ResA, self).__init__()
      self.model = nn.Sequential(# A1
                                nn.BatchNorm2d(C),
                                nn.ReLU(),
                                nn.Conv2d(C,C2,1,stride=1,padding=0,dilation=1,groups=1,bias=True),
                                # B1
                                nn.BatchNorm2d(C2),
                                nn.ReLU(),
                                nn.Conv2d(C2,C2,3,stride=1,padding=1,dilation=1,groups=1,bias=True),
                                # C1
                                nn.BatchNorm2d(C2),
                                nn.ReLU(),
                                nn.Conv2d(C2,C,1,stride=1,padding=0,dilation=1,groups=1,bias=True))
    def forward(self, x):
      res = self.model.forward(x)
      return x + res

# Prints `str` prefixed by the current test function name so we can use it in
# Filecheck label directives.
# This is achieved by inspecting the stack and getting the parent name.
def printWithCurrentFunctionName(s):
  # stack[1] is the caller, i.e. "_test_model"
  # stack[2] is the caller's caller, e.g. "test_conv_1"
  print(inspect.stack()[2][3], s)

class TestMLIRExport(unittest.TestCase):
  def setUp(self):
    pass

  def _test_model(self, model, model_args):
    result = model(model_args)

    mlir = torch_mlir.get_mlir(result)
    printWithCurrentFunctionName (mlir)
    return True

  def test_ResA_16(self):
    dev = torch_mlir.mlir_device()
    model = ResA(16).to(dev)
    passed = self._test_model(model, torch.ones((1,16,128,128), device=dev))
    # CHECK-LABEL: test_ResA_16
    #   CHECK: [[V0:%[a-zA-Z0-9]+]], %{{.*}}, %{{.*}} = "aten.native_batch_norm"({{.*}}) {layer_name = "L0-native_batch_norm-0"} : (tensor<1x16x128x128xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, i1, f32, f32) -> (tensor<1x16x128x128xf32>, tensor<16xf32>, tensor<16xf32>)
    #   CHECK: [[V1:%[a-zA-Z0-9]+]] = "aten.relu"([[V0]]) {layer_name = "L1-relu-0"} : (tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>
    #   CHECK: [[V2:%[a-zA-Z0-9]+]] = "aten.convolution_overrideable"([[V1]], {{.*}}) {layer_name = "L2-convolution_overrideable-0"} : (tensor<1x16x128x128xf32>, {{.*}}) -> tensor<1x8x128x128xf32>
    #   CHECK: [[V3:%[a-zA-Z0-9_]+]], %{{.*}}, %{{.*}} = "aten.native_batch_norm"([[V2]]{{.*}}) {layer_name = "L3-native_batch_norm-1"} : (tensor<1x8x128x128xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, i1, f32, f32) -> (tensor<1x8x128x128xf32>, tensor<8xf32>, tensor<8xf32>)
    #   CHECK: [[V4:%[a-zA-Z0-9]+]] = "aten.relu"([[V3]]) {layer_name = "L4-relu-1"} : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    #   CHECK: [[V5:%[a-zA-Z0-9]+]] = "aten.convolution_overrideable"([[V4]],{{.*}}) {layer_name = "L5-convolution_overrideable-1"} : (tensor<1x8x128x128xf32>, {{.*}}) -> tensor<1x8x128x128xf32>
    #   CHECK: [[V6:%[a-zA-Z0-9_]+]], %{{.*}}, %{{.*}} = "aten.native_batch_norm"([[V5]],{{.*}}) {layer_name = "L6-native_batch_norm-2"} : (tensor<1x8x128x128xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, i1, f32, f32) -> (tensor<1x8x128x128xf32>, tensor<8xf32>, tensor<8xf32>)
    #   CHECK: [[V7:%[a-zA-Z0-9]+]] = "aten.relu"([[V6]]) {layer_name = "L7-relu-2"} : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    #   CHECK: [[V8:%[a-zA-Z0-9]+]] = "aten.convolution_overrideable"([[V7]],{{.*}}) {layer_name = "L8-convolution_overrideable-2"} : (tensor<1x8x128x128xf32>, {{.*}}) -> tensor<1x16x128x128xf32>
    #   CHECK: {{.*}} = "aten.add"(%arg0, [[V8]], {{.*}}) {layer_name = "L9-add-0"} : (tensor<1x16x128x128xf32>, tensor<1x16x128x128xf32>, i32) -> tensor<1x16x128x128xf32>
    self.assertTrue(passed)

verbose = False
if __name__ == '__main__':
    verbose = True
    unittest.main()
