# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

tensor = torch.randn(2,3).to(dev)
result = tensor.t()

mlir = torch_mlir.get_mlir( result )

# CHECK-LABEL: test_export_t
#   CHECK: = "aten.t"(%{{.*}}) {layer_name = "L0-t-0"} : (tensor<2x3xf32>) -> tensor<3x2xf32>
print("test_export_t")
print(mlir)

# CHECK-LABEL: test_jit_t
#   CHECK: PASS!
print("test_jit_t")
ref_result = tensor.to('cpu').t()
if result.to('cpu').equal( ref_result ):
    print ("PASS!")
else:
    print ("fail.")
