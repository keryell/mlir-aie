//===- AIEPasses.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PASSES_H
#define AIE_PASSES_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {

#define GEN_PASS_CLASSES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferAddressesPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEAssignLockIDsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIECanonicalizeDevicePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIECoreToStandardPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEFindFlowsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIELocalizeLocksPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIENormalizeAddressSpacesPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAIERouteFlowsPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIERoutePacketFlowsPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createAIEVectorOptPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEPathfinderPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoStatefulTransformPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoRegisterProcessPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIEUseAIEppEmulationPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

/// Overall Flow:
/// rewrite switchboxes to assign unassigned connections, ensure this can be
/// done concurrently ( by different threads)
/// 1. Goal is to rewrite all flows in the device into switchboxes + shim-mux
/// 2. multiple passes of the rewrite pattern rewriting streamswitch
/// configurations to routes
/// 3. rewrite flows to stream-switches using 'weights' from analysis pass.
/// 4. check a region is legal
/// 5. rewrite stream-switches (within a bounding box) back to flows
struct AIEPathfinderPass : AIERoutePathfinderFlowsBase<AIEPathfinderPass> {

  DynamicTileAnalysis analyzer;

  AIEPathfinderPass() = default;
  AIEPathfinderPass(DynamicTileAnalysis analyzer)
      : analyzer(std::move(analyzer)) {}

  void runOnOperation() override;

  bool attemptFixupMemTileRouting(const mlir::OpBuilder &builder,
                                  SwitchboxOp northSwOp, SwitchboxOp southSwOp,
                                  ConnectOp &problemConnect);

  bool reconnectConnectOps(const mlir::OpBuilder &builder, SwitchboxOp sw,
                           ConnectOp problemConnect, bool isIncomingToSW,
                           WireBundle problemBundle, int problemChan,
                           int emptyChan);

  ConnectOp replaceConnectOpWithNewDest(mlir::OpBuilder builder,
                                        ConnectOp connect, WireBundle newBundle,
                                        int newChannel);
  ConnectOp replaceConnectOpWithNewSource(mlir::OpBuilder builder,
                                          ConnectOp connect,
                                          WireBundle newBundle, int newChannel);

  SwitchboxOp getSwitchbox(DeviceOp &d, int col, int row);
};

} // namespace xilinx::AIE

#endif // AIE_PASSES_H
