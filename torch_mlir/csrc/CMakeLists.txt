
# This command generates most of the boilerplate definitions of the ATen
# Dialect from the corresponding pytorch exports.
add_custom_command(OUTPUT ATenMLIRType_default.cpp ATenMLIRType_default.h ATenOps.td
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../gen_aten_dialect.py --output_folder=${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/ATenMLIRType.h ${TORCH_INSTALL_PREFIX}/include/ATen/RegistrationDeclarations.h ${TORCH_INSTALL_PREFIX}/include/ATen/Functions.h
  DEPENDS ATenMLIRType.h
  )
add_custom_target(aten_type_gen DEPENDS ATenMLIRType_default.cpp ATenMLIRType_default.h)


include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_INSTALL_PREFIX}/include/TH
  ${TORCH_INSTALL_PREFIX}/include/THC/opt/pytorch/pytorch
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PYTHON_INCLUDE_DIRS}
  )
link_directories("${TORCH_INSTALL_PREFIX}/lib")
add_library(_torch_mlir SHARED
  ATenMLIRBridge.cpp
  ATenMLIRType.cpp
  ATenMLIRType_default.cpp
  ATenMLIRDevice.cpp
  InitPythonBindings.cpp
  ir.cpp
  jit.cpp
  MLIRGenerator.cpp
  Tensor.cpp
  TensorImpl.cpp
  TorchUtil.cpp
  )
set_target_properties(_torch_mlir PROPERTIES PREFIX "")

get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
target_link_libraries(_torch_mlir
  NPCOMPATenDialect
  ${TORCH_LIBRARIES}
  ${mlir_libs}
  ${PYTHON_LIBRARIES}
  torch_python
  )
add_dependencies(_torch_mlir aten_type_gen)
