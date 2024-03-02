//===- AIEUseAIEppEmulation.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// #include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/TargetParser/Host.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "aie-use-aie++-emulation"

namespace xilinx::AIE {

namespace {

/// Replace the AIE intrisics calls to call to AIE++ emulator functions mangled
/// with column and row so they are unique
void rewriteAIEIntrinsics(mlir::ModuleOp module, mlir::func::FuncOp f,
                          llvm::StringRef col, llvm::StringRef row) {
  constexpr auto AIEPP_PUT_MS_FORMAT{
      "_Z25aie_tile_put_ms_intrinsicITnDaLi{0}ETnDaLi{1}EEvii"};
  f->walk([&](mlir::func::CallOp c) {
    if (c.getCallee() == "llvm.aie.put.ms") {
      std::string aieppIntrinsicsName =
          llvm::formatv(AIEPP_PUT_MS_FORMAT, col, row);
      if (auto aieppIntrinsics =
              module.lookupSymbol<mlir::func::FuncOp>(aieppIntrinsicsName);
          !aieppIntrinsics) {
        // Declare the intrinsics at the end of the module so the reader can
        // focus on the important part first
        auto b = mlir::OpBuilder::atBlockEnd(module.getBody());
        aieppIntrinsics = b.create<mlir::func::FuncOp>(
            b.getUnknownLoc(), aieppIntrinsicsName,
            mlir::FunctionType::get(b.getContext(),
                                    {b.getI32Type(), b.getI32Type()}, {}));
        aieppIntrinsics.setPrivate();
      }
      c.setCallee(aieppIntrinsicsName);
    }
  });
}

/// Rewrite the core tile program functions to target AIE++
///
/// Rename the core tile program functions and AIE intrisics to have a mangled
/// name compatible with AIE++ emulator.
void rewriteTileProgramFunctionsForEmulation(mlir::ModuleOp module) {
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
      rewriteAIEIntrinsics(module, f, col, row);
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

    rewriteTileProgramFunctionsForEmulation(module);
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEUseAIEppEmulationPass() {
  return std::make_unique<AIEUseAIEppEmulationPass>();
}

} // namespace xilinx::AIE
