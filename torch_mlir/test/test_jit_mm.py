# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

t0 = torch.randn((3,13), device=dev)
t1 = torch.randn((13,5), device=dev)
print(t0.to('cpu'), t1.to('cpu'))
print(torch.mm(t0.to('cpu'), t1.to('cpu')))

t2 = torch.mm(t0, t1)
#
# Generate and check the MLIR for the result tensor
#
t2_mlir = torch_mlir.get_mlir( t2 )

# CHECK-LABEL: test_jit_mul2_export
#   CHECK: %0 = "aten.mm"(%arg0, %arg1) {layer_name = "L0-mm-0"} : (tensor<3x13xf32>, tensor<13x5xf32>) -> tensor<3x5xf32>
print("test_jit_mul2_export")
print(t2_mlir)

#
# Check the result tensor against the CPU
#
t0_cpu = t0.to('cpu')
t1_cpu = t1.to('cpu')
t2_cpu = t2.to('cpu')

print (t0_cpu, " *\n", t1_cpu, " =\n", t2_cpu)

# CHECK-LABEL: test_jit_mm
#   CHECK: PASS!
print("test_jit_mm")
ref_tensor = torch.mm(t0_cpu, t1_cpu)
error = (ref_tensor - t2_cpu).abs().max()
if error <= 1e-5:
    print ("PASS!")
else:
    print ("fail.")
