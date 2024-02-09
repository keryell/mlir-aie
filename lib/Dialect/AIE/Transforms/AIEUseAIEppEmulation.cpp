//===- AIEUseAIEppEmulation.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "aie-use-aie++-emulation"

namespace xilinx::AIE {

struct AIEUseAIEppEmulationPass
    : AIEUseAIEppEmulationBase<AIEUseAIEppEmulationPass> {
  void runOnOperation() override {
    // DeviceOp device = getOperation();
  }
};

std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEUseAIEppEmulationPass() {
  return std::make_unique<AIEUseAIEppEmulationPass>();
}

} // namespace xilinx::AIE
