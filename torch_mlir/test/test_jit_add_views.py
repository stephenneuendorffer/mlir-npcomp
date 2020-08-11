# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()
t0 = torch.randn((4,16,4), device=dev)
t1 = torch.randn((4,16,4), device=dev)

t3 = torch.randn((4,64), device=dev)
t4 = torch.randn((4,64), device=dev)

t2 = t0 + t1
t5 = t3 + t4

t6 = t5.view((4,4,4,4))
t7 = t2.view((4,4,4,4))

t8 = t6 + t7

t0_cpu = t0.to('cpu')
t1_cpu = t1.to('cpu')
t2_cpu = t2.to('cpu')

# CHECK-LABEL: test_jit_add_views_0
#   CHECK: PASS!
print("test_jit_add_views_0")
if t2_cpu.equal(t0_cpu + t1_cpu):
    print ("PASS!")
else:
    print ("fail.")

t3_cpu = t3.to('cpu')
t4_cpu = t4.to('cpu')
t5_cpu = t5.to('cpu')

# CHECK-LABEL: test_jit_add_views_1
#   CHECK: PASS!
print("test_jit_add_views_1")
if t5_cpu.equal(t3_cpu + t4_cpu):
    print ("PASS!")
else:
    print ("fail.")

t6_cpu = t6.to('cpu')
t7_cpu = t7.to('cpu')
t8_cpu = t8.to('cpu')

# CHECK-LABEL: test_jit_add_views_2
#   CHECK: PASS!
print("test_jit_add_views_2")
if t8_cpu.equal(t6_cpu + t7_cpu):
    print ("PASS!")
else:
    print ("fail.")