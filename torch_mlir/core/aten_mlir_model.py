# -*- Python -*-
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch_mlir

def get_mlir_supported_devices(devkind=None):
  # FIXME: how do we define our own device?
  return ["xla:0"]

def mlir_device(devkind=None):
  devices = get_mlir_supported_devices(devkind=devkind)
  device = devices[0]
  return torch.device(device)
