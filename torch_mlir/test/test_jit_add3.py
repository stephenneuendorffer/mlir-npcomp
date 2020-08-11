# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()
t0 = torch.randn((1,2,3,4), device=dev)
t1 = torch.randn((1,2,3,4), device=dev)
t2 = torch.randn((1,2,3,4), device=dev)

t3 = t0 + t1 + t2

#
# Generate and check the MLIR for the result tensor
#
t3_mlir = torch_mlir.get_mlir( t3 )

# CHECK-LABEL: test_jit_add3_export
#   CHECK: %1 = "aten.add"(%arg0, %arg1, %0) {layer_name = "L0-add-0"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, i32) -> tensor<1x2x3x4xf32>
#   CHECK: %2 = "aten.add"(%1, %arg2, %0) {layer_name = "L1-add-1"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, i32) -> tensor<1x2x3x4xf32>
print("test_jit_add3_export")
print(t3_mlir)

#
# Check the result tensor against the CPU
#
t0_cpu = t0.to('cpu')
t1_cpu = t1.to('cpu')
t2_cpu = t2.to('cpu')
t3_cpu = t3.to('cpu')

print (t0_cpu, " +\n", t1_cpu, " +\n", t2_cpu, " =\n", t3_cpu)

# CHECK-LABEL: test_jit_add3_run
#   CHECK: PASS!
print("test_jit_add3_run")
if t3_cpu.equal(t0_cpu + t1_cpu + t2_cpu):
    print ("PASS!")
else:
    print ("fail.")
