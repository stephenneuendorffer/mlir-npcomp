//===- TorchUtil.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace torch_mlir {
namespace util {

at::Tensor Zeros(at::IntArrayRef sizes, at::ScalarType type);

// Makes a deep copy of an ATEN tensor.
at::Tensor CopyTensor(const at::Tensor &ref);

// Same as above, with an additional cast.
at::Tensor CopyTensor(const at::Tensor &ref, at::ScalarType dest_type);

// Return at::ScalarType from at::Scalar
at::ScalarType GetScalarType(at::Scalar scalar);

template <typename T, typename S>
T OptionalOr(const c10::optional<S> &value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}

} // namespace util
} // namespace torch_mlir
