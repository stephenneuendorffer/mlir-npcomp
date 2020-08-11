# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

#torch_mlir.set_debug(True)
# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

x = torch.rand((3,64,8,8), device=dev)
y = x*x
print (y.stride())

dim = [64,24,24]
dim = [4,4,4]
N = 2;
count = dim[0]*dim[1]*dim[2]
sizes = (N,dim[0],dim[1],dim[2])
strides = (1,dim[1]*dim[2],dim[2],1)
print(count)
t0 = torch.randn((N,count), device=dev)
t0_like = torch.randn((N,count))


t1 = t0.as_strided(sizes, strides)
t1_ref = t0.to('cpu').as_strided(sizes, strides)
t1_like = t0_like.as_strided(sizes, strides)

t1_ref = t1_ref.clone()

t1_mlir = torch_mlir.get_mlir( t1 )
print(t1_mlir)

# check that the IR has recorded the
# stride properly before invoking JIT
#   CHECK: PASS!
print("test_unjitted_stride")
if (t1.stride() == t1_like.stride()):
    print ("PASS!")
else:
    print ("fail.")

# CHECK-LABEL: test_jit_as_stride
#   CHECK: PASS!
#   CHECK: PASS!
print("test_jit_as_stride")
cpu = t1.to('cpu')
if t1_ref.equal(cpu):
    print ("PASS!")
else:
    print ("fail.")
    for i in range(0, count-1):
        print("@", i, t0.to("cpu")[i])
        r = t1_ref[i]
        t = cpu[i]
        if not r.equal(t):
            print ("t1_ref", r)
            print ("cpu", t)
        # for a in range(0, dim[0]-1):
        #     for b in range(0, dim[1]-1):
        #         for c in range(0, dim[2]-1):
        #             r = t1_ref[i][a][b][c]
        #             t = cpu[i][a][b][c]
        #             if not r.equal(t):
        #                 print ("@", a, ",", b, ",", c)
        #                 print ("t1_ref", r)
        #                 print ("cpu", t)

if t1_ref.stride() == cpu.stride():
    print ("PASS!")
else:
    print ("fail.")
