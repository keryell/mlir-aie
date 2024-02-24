//===- AIEUseAIEppEmulation.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <regex>

// #include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/TargetParser/Host.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "aie-use-aie++-emulation"

namespace xilinx::AIE {

namespace {

void renameTileProgramFunctionsForEmulation(mlir::ModuleOp module) {
  // The name format of a function representing a tile program
  static const llvm::Regex CORE_NAME{"^core_([[:digit:]]+)_([[:digit:]]+)$"};
  // The name format of the function inside AIE++ emulator to be formatted
  // with col, row parameters
  constexpr auto AIEPP_EMULATOR_FORMAT{
      "_Z8air_tileITnDaLi{0}ETnDaLi{1}EEvPvPFvS0_jE"};

  module->walk([&](mlir::func::FuncOp f) {
    if (llvm::SmallVector<llvm::StringRef> matches;
        CORE_NAME.match(f.getName(), &matches)) {
      const auto &col = matches[1];
      const auto &row = matches[2];
      std::string name = llvm::formatv(AIEPP_EMULATOR_FORMAT, col, row);
      f.setName(name);
    }
  });
}

} // namespace

struct AIEUseAIEppEmulationPass
    : AIEUseAIEppEmulationBase<AIEUseAIEppEmulationPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    auto builder = mlir::OpBuilder::atBlockEnd(module.getBody());

    // Set the target triple as the same as the machine running this right now
    // to run the emulation on a similar machine later
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    builder.getStringAttr(llvm::sys::getDefaultTargetTriple()));

    renameTileProgramFunctionsForEmulation(module);
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEUseAIEppEmulationPass() {
  return std::make_unique<AIEUseAIEppEmulationPass>();
}

} // namespace xilinx::AIE
