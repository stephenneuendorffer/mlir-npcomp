//===- ATenMLIRDevice.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Based off code from git@github.com:pytorch/xla.git

#include "ATenMLIRDevice.h"

namespace torch_mlir {
namespace {

std::string DeviceTypeToString(DeviceType hw_type) {
  switch (hw_type) {
  case DeviceType::CPU:
    return "CPU";
  case DeviceType::MLIR:
    return "MLIR";
  }
  return "";
}

void ParseDevice(const std::string &device_spec, Device *device) {
  if (device_spec.empty()) {
    return ParseDevice(std::string("acap:0"), device);
  }

  if (device_spec[0] == ':') {
    return ParseDevice(std::string("acap") + device_spec, device);
  }

  auto pos = device_spec.find(':');
  auto devtype = device_spec.substr(0, pos);

  // TODO error check

  device->ordinal =
      std::stoi(device_spec.substr(pos + 1, device_spec.size() - pos - 1));
  if (devtype == "MLIR") {
    device->hw_type = DeviceType::MLIR;
  } else if (devtype == "CPU") {
    device->hw_type = DeviceType::CPU;
  } else {
    // TODO, error
    device->hw_type = DeviceType::MLIR;
  }
}

} // namespace

Device::Device(const std::string &device_spec) {
  ParseDevice(device_spec, this);
}

std::string Device::ToString() const {
  return DeviceTypeToString(hw_type) + std::string(":") +
         std::to_string(ordinal);
}

const Device *GetDefaultDevice() {
  static const Device *default_device = new Device("");
  return default_device;
}

} // namespace torch_mlir
