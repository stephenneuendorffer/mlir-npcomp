//===- ATenMLIRBridge.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Based off code from git@github.com:pytorch/xla.git

// This file implements a bridge which moves data back and forth from torch
// tensors (at::Tensor) to MLIRTensor, which represents a tensor associated
// with our virtual 'MLIR' device.

#include "ATenMLIRDevice.h"
#include "Tensor.h"

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

namespace torch_mlir {
namespace bridge {

c10::optional<MLIRTensor> TryGetMLIRTensor(const at::Tensor &tensor);

// Return an MLIR tensor that is computed the same way as the given at::Tensor
MLIRTensor GetMLIRTensor(const at::Tensor &tensor);

MLIRTensor GetOrCreateMLIRTensor(const at::Tensor &tensor,
                                 const Device &device);

// Creates a vector of at::Tensor objects extracted from a list of MLIR tensors.
std::vector<at::Tensor> MLIRCreateTensorList(const at::TensorList &tensors);

c10::optional<Device> GetMLIRDevice(const at::TensorList &tensors);

c10::optional<Device> GetMLIRDevice(const at::TensorOptions &tensor_options);

c10::optional<Device> GetMLIRDevice(const c10::Device &device);

c10::optional<Device> GetMLIRDevice(const at::Tensor &tensor);

Device AtenDeviceToMLIRDevice(const c10::Device &device);

c10::Device MLIRDeviceToAtenDevice(const Device &device);

at::Tensor MLIRToAtenTensor(MLIRTensor device_tensor,
                            const at::TensorOptions &tensor_options);

// Create an Aten tensor with MLIR type id from MLIRTensor
at::Tensor AtenFromMLIRTensor(MLIRTensor device_tensor);

// Creates an MLIR tensor holding the data in tensor, on the given device.
at::Tensor CreateMLIRTensor(at::Tensor tensor,
                            const c10::optional<Device> &device);

} // namespace bridge

} // namespace torch_mlir
