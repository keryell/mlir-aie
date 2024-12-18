//===- aie-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef CLANGIR_MLIR_FRONTEND
#include "aie/CIR/CIRToAIEPasses.h"
#endif
#include "aie/Conversion/Passes.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/Analysis/Passes.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/TransformOps/DialectExtension.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"
#include "aie/version.h"
#ifdef CLANGIR_MLIR_FRONTEND
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/LowerToMLIR.h"
#include "clang/CIR/Passes.h"
#endif

static void versionPrinter(llvm::raw_ostream &os) {
  os << "aie-opt " << AIE_GIT_COMMIT << "\n";
}

int main(int argc, char **argv) {

  mlir::registerAllPasses();
  xilinx::registerConversionPasses();
  xilinx::AIE::registerAIEPasses();
  xilinx::AIEX::registerAIEXPasses();
  xilinx::aievec::registerAIEVecAnalysisPasses();
  xilinx::aievec::registerAIEVecPasses();
  xilinx::aievec::registerAIEVecPipelines();
#ifdef CLANGIR_MLIR_FRONTEND
  xilinx::AIE::CIR::registerCIRToAIEPasses();
#endif

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  xilinx::registerAllDialects(registry);

  registerAllExtensions(registry);

  xilinx::aievec::registerTransformDialectExtension(registry);

  llvm::cl::AddExtraVersionPrinter(versionPrinter);

#ifdef CLANGIR_MLIR_FRONTEND
  // ClangIR dialect
  registry.insert<cir::CIRDialect>();

  // ClangIR-specific passes
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return cir::createConvertMLIRToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRCanonicalizePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createCIRSimplifyPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createLifetimeCheckPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createDropASTPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createLoweringPreparePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createSCFPreparePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createHoistAllocasPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createGotoSolverPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createIdiomRecognizerPass();
  });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return mlir::createLibOptPass(); });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createCallConvLoweringPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return cir::createConvertCIRToMLIRPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return cir::createConvertCIRToMLIRPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return cir::direct::createConvertCIRToLLVMPass();
  });

  mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions> pipeline(
      "cir-to-llvm", "Full pass pipeline from CIR to LLVM MLIR dialect",
      [](mlir::OpPassManager &pm) {
        cir::direct::populateCIRToLLVMPasses(pm, /* useCCLowering */ true);
      });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createFlattenCFGPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });

  mlir::registerTransformsPasses();

  cir::runAtStartOfConvertCIRToMLIRPass([](mlir::ConversionTarget ct) {
    ct.addLegalDialect<xilinx::AIE::AIEDialect, xilinx::AIEX::AIEXDialect,
                       xilinx::aievec::aie1::AIEVecAIE1Dialect,
                       xilinx::aievec::AIEVecDialect>();
    ct.addLegalOp<mlir::UnrealizedConversionCastOp>();
  });

  cir::direct::runAtStartOfConvertCIRToLLVMPass([](mlir::ConversionTarget ct) {
    ct.addLegalDialect<xilinx::AIE::AIEDialect, xilinx::AIEX::AIEXDialect,
                       xilinx::aievec::aie1::AIEVecAIE1Dialect,
                       xilinx::aievec::AIEVecDialect>();
    ct.addLegalOp<mlir::UnrealizedConversionCastOp>();
  });
#endif

  return failed(
      MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry));
}
