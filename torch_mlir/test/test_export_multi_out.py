# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

t0 = torch.randn(4, device=dev)
t1 = torch.randn(4, device=dev)
t2 = torch.randn(4, device=dev)

t4 = t0 + t1 + t2
t5 = t4 + t1
t6 = t5 + t4

# CHECK-LABEL: test_multi_out
#   CHECK: return %2, %3, %4 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
mlir = torch_mlir.get_mlir([t4, t5, t6])
print ("test_multi_out")
print (mlir)
