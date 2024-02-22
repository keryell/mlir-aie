//===- AIEUseAIEppEmulation.cpp ---------------------------------*- C++ -*-===//
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

#include "llvm/TargetParser/Host.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "aie-use-aie++-emulation"

namespace xilinx::AIE {

struct AIEUseAIEppEmulationPass
    : AIEUseAIEppEmulationBase<AIEUseAIEppEmulationPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    auto builder = mlir::OpBuilder::atBlockEnd(module.getBody());

    // Set the target triple as the same as the machine running this right now
    // to run the emulation on a similar machine later
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    builder.getStringAttr(llvm::sys::getDefaultTargetTriple()));
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEUseAIEppEmulationPass() {
  return std::make_unique<AIEUseAIEppEmulationPass>();
}

} // namespace xilinx::AIE
