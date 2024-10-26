//===- CIRToAIEpasses.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//

#include <any>
#include <array>
#include <cassert>

#include "aie/CIR/CIRToAIEPasses.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToMLIR.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using namespace std::string_literals;

namespace xilinx::AIE::CIR {

struct CIRToAIETypesAnalysis {
  // llvm::DenseMap<mlir::Type, std::optional<mlir::Type>> types;
  struct AIELikeTypesDeconstruction {
    // For example "aie::device<aie::npu1>"
    std::string fullName;
    // For example "aie::device"
    std::string base;
    // For example "npu1"
    std::vector<std::string> subMatches;
    // To attach something, like the aie.tile operation for example
    std::any data;

    std::string str() {
      return "Fullname = " + fullName + ", base = " + base +
             ", subMatches = " + llvm::join(subMatches, ", ");
    }
  };

  // A map from a type do its aie:: deconstruction in the case it is a pointer
  // type to a well known aie:: struct
  llvm::DenseMap<mlir::Type, std::optional<AIELikeTypesDeconstruction>>
      moduleTypes;

  void analyze() {
    // A struct with a name like "aie::device<aie::npu1>" (and the "npu1" is
    // used directly for the MLIR aie.device attribute) or aie::tile_t<8,50> for
    // example
    static const std::array typeNamePatterns{
        llvm::Regex{"^(aie::device)<aie::([^>]+)>$"},
        llvm::Regex{"^(aie::tile)<([[:digit:]]+), ([[:digit:]]+)>$"},
        llvm::Regex{"^(aie::buffer)<([^,]+), ([^>]+)>$"}};

    for (auto &[type, value] : moduleTypes) {
      if (auto maybePointerType = mlir::dyn_cast<mlir::cir::PointerType>(type))
        if (auto maybeStructType = mlir::dyn_cast<mlir::cir::StructType>(
                maybePointerType.getPointee()))
          for (auto &tnp : typeNamePatterns)
            if (llvm::SmallVector<llvm::StringRef> matches;
                tnp.match(maybeStructType.getName(), &matches)) {
              value = {.fullName = matches[0].str(), .base = matches[1].str()};
              for (auto &e : llvm::ArrayRef(matches.begin() + 2, matches.end()))
                value->subMatches.emplace_back(e.str());
              // No need to look for a next match, go for the next type to
              // categorize
              break;
            }
    }
  }

public:
  CIRToAIETypesAnalysis(mlir::ModuleOp module) {
    module->walk([this](mlir::Operation *op) {
      for (auto result : op->getResults()) {
        auto type = result.getType();
        moduleTypes.try_emplace(type, std::nullopt);
      }
    });
    analyze();
  }

  void dump() {
    for (auto &[type, value] : moduleTypes) {
      llvm::outs() << "Type: " << type << " value: ";
      if (value) {
        llvm::outs() << value->str() << '\n';
      } else
        llvm::outs() << "None\n";
    }
  }
};

namespace {

// Return true if the call operation calls a function with any of the given
// string annotations
bool isCallingFunctionWithAnnotation(
    mlir::cir::CallOp op, llvm::ArrayRef<llvm::StringRef> anyAnnotations) {
  if (auto calledFunc =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
              op, op.getCalleeAttr())) {
    if (auto annnotations = calledFunc.getAnnotationsAttr())
      for (auto a : calledFunc.getAnnotationsAttr()) {
        for (auto one : anyAnnotations)
          if (mlir::cast<mlir::cir::AnnotationAttr>(a).getName() == one)
            return true;
      }
  }
  return false;
}

// Return true if the UnrealizedConversionCast operation has any of the given
// string annotations
bool isUnrealizedConversionCastWithAnnotation(
    mlir::UnrealizedConversionCastOp op,
    llvm::ArrayRef<llvm::StringRef> anyAnnotations) {
  for (auto attr : op->getAttrDictionary())
    for (auto needle : anyAnnotations)
      if (attr.getName() == needle)
        return true;
  return false;
}

// Generate the equivalent memref type of an aie::buffer
mlir::MemRefType bufferMemrefType(mlir::Type buffer,
                                  mlir::ConversionPatternRewriter &rewriter) {
  static mlir::TypeConverter typeConverter = cir::prepareTypeConverter();
  buffer.dump();
  if (auto p = mlir::dyn_cast<mlir::cir::PointerType>(buffer)) {
    if (auto bufferType =
            mlir::dyn_cast<mlir::cir::StructType>(p.getPointee())) {
      bufferType.dump();
      // For now the aie::buffer is implemented as a std::array in the buffer
      // struct
      auto members = bufferType.getMembers();
      if (auto stdArrayType =
              mlir::dyn_cast<mlir::cir::StructType>(members.front())) {
        stdArrayType.dump();
        // Access the array inside the std::array struct
        if (auto arrayType = mlir::dyn_cast<mlir::cir::ArrayType>(
                stdArrayType.getMembers().front())) {
          arrayType.dump();
          auto memref = mlir::dyn_cast<mlir::MemRefType>(
              typeConverter.convertType(arrayType));
          memref.dump();
          return memref;
        }
      }
    }
  }
  return {};
}

// Lower C++ code like \code aie::device<aie::npu1> into an \code
// aie.device(npu1){} operation
struct PrepareDeviceLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
  using mlir::OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // The struct has a name like "aie::device<aie::npu1>" and the "npu1"
    // is used directly for the MLIR aie.device attribute
    if (auto aieLike = cat->moduleTypes[op.getType()];
        aieLike && aieLike->base == "aie::device") {
      auto deviceName = aieLike->subMatches[0];
      auto deviceId =
          xilinx::AIE::symbolizeEnum<xilinx::AIE::AIEDevice>(deviceName);
      if (!deviceId)
        // Actually this test cannot happens since the API of
        // xilinx::AIE::symbolizeEnum is strange: even if it returns a
        // std::optional it errors without returning
        op.emitError() << "aie::device incorrect for '" << deviceName << "'";
      // Replace the alloca of the aie::device by a temporary cast from
      // thin air and add a named attribute to the device name to make things
      // clearer
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          op, op.getResult().getType(), mlir::ValueRange{},
          std::array{rewriter.getNamedAttr(
              aieLike->base, rewriter.getAttr<mlir::StringAttr>(deviceName))});
      return mlir::success();
    }
    return mlir::failure();
  }
};

// clang-format off
// Rewrite something like
//    %2 = cir.alloca !ty_aie3A3Atile3C12C_43E, !cir.ptr<!ty_aie3A3Atile3C12C_43E>, ["t", init] {alignment = 1 : i64} loc(#loc102)
//    %4 = cir.call @_ZN3aie6deviceILNS_3$_0E42EE4tileILi1ELi4EEENS_4tileIXT_EXT0_EEEv(%1) : (!cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu13E>) -> !ty_aie3A3Atile3C12C_43E loc(#loc70)
//    cir.store %4, %2 : !ty_aie3A3Atile3C12C_43E, !cir.ptr<!ty_aie3A3Atile3C12C_43E> loc(#loc70)
//
// Into
//
//    %2 = builtin.unrealized_conversion_cast %1 : !cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu13E> to !cir.ptr<!ty_aie3A3Atile3C12C_43E> {"aie::tile" = ["1", "4"]}
// clang-format on
struct PrepareTileBufferLowering
    : public mlir::OpConversionPattern<mlir::cir::CallOp> {
  using mlir::OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  // \todo Find a less ugly way to access the analysis. How is it possible for a
  // pattern to access some contextual information?
  // It should be OK since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (isCallingFunctionWithAnnotation(
            op, {"aie.device.tile", "aie.tile.buffer"})) {
      auto device = op.getOperand(0);
      auto user = op.getResult().getUsers().begin();
      // Track the alloca where the tiled is stored
      auto store = mlir::dyn_cast<mlir::cir::StoreOp>(*user);
      auto alloca = mlir::dyn_cast<mlir::cir::AllocaOp>(
          store.getOperand(1).getDefiningOp());
      if (auto aieLike = cat->moduleTypes[alloca.getResult().getType()];
          aieLike) {
        // Replace the alloca by a conversion to be replaced later in
        // another pass.
        // Keep analyzed type information as named attribute to make things
        // clearer
        llvm::SmallVector<mlir::Attribute, 4> attrs;
        for (auto e : aieLike->subMatches)
          attrs.emplace_back(rewriter.getAttr<mlir::StringAttr>(e));
        rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
            alloca, alloca.getResult().getType(), device,
            std::array{rewriter.getNamedAttr(aieLike->base,
                                             rewriter.getArrayAttr(attrs))});
        // Remove the now useless original operations
        rewriter.eraseOp(store);
        rewriter.eraseOp(op);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

/*
  Replace the call to

  cir.func internal private
 @_ZN3aie6tile_tILi1ELi4EE7programIZ4mainE3$_0EEvOT_(%arg0:
 !cir.ptr<!ty_aie3A3Atile_t3C12C_43E>, %arg1: !cir.ptr<!ty_anon2E0_>)
 [#cir.annotation<name = "aie.tile.program", args = []>] extra(#fn_attr)

 which ends up calling the lambda

 cir.call @_ZZ4mainENK3$_0clEv(%5) : (!cir.ptr<!ty_anon2E0_>) -> ()

 by just inlining the lambda body into the aie.core operation and replacing the
 capture by the direct def/use forwarding

*/
struct PrepareCoreLowering
    : public mlir::OpConversionPattern<mlir::cir::CallOp> {
  using mlir::OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (isCallingFunctionWithAnnotation(op, {"aie.tile.program"})) {
      // Get tile::program() member function
      if (auto calledFunc =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                  op, op.getCalleeAttr())) {
        // The last function instruction is cir.return and the one before
        // is the call to the lambda
        // calledFunc.getBlocks().front().back().dump();
        auto lambdaCall = mlir::dyn_cast<mlir::cir::CallOp>(
            *std::next(calledFunc.getBlocks().front().rbegin()));
        // lambdaCall.dump();
        if (auto lambdaFunc =
                mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                    lambdaCall, lambdaCall.getCalleeAttr())) {
          // lambdaFunc.dump();
          assert(lambdaFunc.getLambda());
          // auto scopeOp = op->getParentOfType<mlir::cir::ScopeOp>();
          // scopeOp.dump();
          //  The aie++ tile value
          rewriter.setInsertionPoint(op);
#if 0
          auto tile = op.getOperand(0);
          auto pseudoTile = rewriter.create<mlir::UnrealizedConversionCastOp>(
              op.getLoc(), rewriter.getIndexType(), tile);
          pseudoTile.dump();
          // Cannot create a CoreOp here because the verifyer will expect some
          // AIE neighborhood
          rewriter.create<xilinx::AIE::CoreOp>(op.getLoc(),
                                               pseudoTile.getOutputs().front());
#endif
          rewriter.eraseOp(op);
          //        rewriter.insert(coreOp);
          // coreOp.dump();

          // auto bs = lambdaFunc.getBlocks().begin();
          //          rewriter.inlineBlockBefore(Block *source, Block *dest,
          //          Block::iterator before)
          return mlir::success();
        }
      }
    }

    return mlir::failure();
  }
};

struct CIRToAIEPrepare : CIRToAIEPrepareBase<CIRToAIEPrepare> {
  void runOnOperation() override {
    // Compute the analysis for the module since it is a module pass.
    // \todo Should this be a real pass?
    auto &cat = getAnalysis<CIRToAIETypesAnalysis>();
    // \todo Clean up this mess
    PrepareDeviceLowering::cat = &cat;
    PrepareTileBufferLowering::cat = &cat;
    // See mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
    mlir::ConversionTarget target{getContext()};
    target.addLegalDialect<xilinx::AIE::AIEDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<mlir::cir::AllocaOp>(
        [&](mlir::cir::AllocaOp op) {
          // If the struct has a name like "aie::device<aie::npu1>", mark
          // the operation illegal so it has to be rewritten
          auto aieLike = cat.moduleTypes[op.getType()];
          return !(aieLike && aieLike->base == "aie::device");
        });
    target.addDynamicallyLegalOp<mlir::cir::CallOp>([](mlir::cir::CallOp op) {
      return !isCallingFunctionWithAnnotation(
          op, {"aie.device.tile", "aie.tile.buffer"});
    });
    mlir::RewritePatternSet patterns{&getContext()};
    patterns.add<PrepareDeviceLowering>(&getContext());
    patterns.add<PrepareTileBufferLowering>(&getContext());
    //    patterns.add<PrepareCoreLowering>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

// Erase a range of Operation* and its users recursively
template <typename OpRange>
void eraseOpsAndUsers(OpRange &&opsToErase) {
  llvm::SetVector<mlir::Operation *> allOpsAndUsers;
  llvm::SmallVector<mlir::Operation *> newOps{
      std::forward<OpRange>(opsToErase)};
  // While there are some operations to process
  while (!newOps.empty()) {
    auto *op = newOps.pop_back_val();
    // If the operation has not been visited yet, add it to the set and process
    // its users
    if (allOpsAndUsers.insert(op))
      for (auto result : op->getResults())
        for (auto *user : result.getUsers())
          // Add each user to the visit queue
          newOps.push_back(user);
  }
  // To avoid erasing operations with some users, topologically sort the
  // operations according to their use-def chains and erase them in reverse
  // order
  mlir::topologicalSort(allOpsAndUsers);
  for (auto *op : llvm::reverse(allOpsAndUsers))
    op->erase();
}

// Lower aie::tile::program(<tile code>)
void lowerTileProgram(xilinx::AIE::TileOp t, mlir::cir::CallOp c,
                      mlir::ConversionPatternRewriter &rewriter) {
  if (auto calledFunc =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
              c, c.getCalleeAttr())) {
    // The last function instruction is cir.return and the one before
    // is the call to the lambda
    // calledFunc.getBlocks().front().back().dump();
    auto lambdaCall = mlir::dyn_cast<mlir::cir::CallOp>(
        *std::next(calledFunc.getBlocks().front().rbegin()));
    lambdaCall.emitRemark("lambdaCall");
    if (auto lambdaFunc =
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::cir::FuncOp>(
                lambdaCall, lambdaCall.getCalleeAttr())) {
      lambdaFunc.emitRemark("Tile core lambda");
      assert(lambdaFunc.getLambda());
      auto scopeOp = c->getParentOfType<mlir::cir::ScopeOp>();
      scopeOp.emitRemark("Scope");
      // Create the core op at the end of the block owning the tile op, which is
      // inside the device op region
      rewriter.setInsertionPointToEnd(t->getBlock());
      auto coreOp = rewriter.create<xilinx::AIE::CoreOp>(c.getLoc(), t);
      // Create the empty block of the core op region
      coreOp.getRegion().emplaceBlock();
      coreOp.emitRemark("Brand-new core");
      mlir::UnrealizedConversionCastOp ucc;
      scopeOp.getRegion().front().walk<mlir::WalkOrder::PreOrder>(
          [&](mlir::UnrealizedConversionCastOp u) {
            u.emitRemark("Found UnrealizedConversionCastOp inside the scope");
            ucc = u;
            return mlir::WalkResult::interrupt();
          });
      // Values can be replaced while cloning, not operations
      mlir::IRMapping irm;
      // Connect the input of old cast inside the clone to the output of tile op
      irm.map(ucc.getOperand(0), t.getResult());
      rewriter.setInsertionPointToEnd(&coreOp.getRegion().front());
      auto *clone = rewriter.clone(*scopeOp.getOperation(), irm);
      clone->emitRemark("Clone");
      coreOp.emitRemark("Stuffed core");
      scopeOp.emitRemark("Scope after cloning");
    }
  }
}

struct DeviceLoweringRP
    : public mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp> {
  using mlir::OpConversionPattern<
      mlir::UnrealizedConversionCastOp>::OpConversionPattern;

  // \todo Find a less ugly way to access the analysis. How is it possible
  // for a pattern to access some contextual information? It should be OK
  // since it is a module pass, so no parallelism here.
  static inline CIRToAIETypesAnalysis *cat;

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    op.emitRemark("DeviceLowering");
    if (auto aieLike = cat->moduleTypes[op.getType(0)];
        aieLike && aieLike->base == "aie::device") {
      auto deviceName = aieLike->subMatches[0];
      auto deviceId =
          xilinx::AIE::symbolizeEnum<xilinx::AIE::AIEDevice>(deviceName);
      if (!deviceId)
        // Actually this test cannot happens since the API of
        // xilinx::AIE::symbolizeEnum is strange: even if it returns a
        // std::optional it errors without returning
        op.emitError("aie::device incorrect for '") << deviceName << "'";
      rewriter.setInsertionPoint(op);
      auto deviceOp =
          rewriter.create<xilinx::AIE::DeviceOp>(op.getLoc(), *deviceId);
      // The aie.device requires one block
      deviceOp.getRegion().emplaceBlock();
      // Create all the following code inside the device region
      rewriter.setInsertionPointToStart(deviceOp.getBody());
      for (mlir::Operation *user : op.getResult(0).getUsers()) {
        if (auto tileCast =
                mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user)) {
          tileCast.emitRemark("tileCast from device");
          if (auto t = cat->moduleTypes[tileCast.getType(0)]) {
            // llvm::errs() << t->str() << " value \n";
            auto col = t->subMatches[0];
            // llvm::errs() << "col" << col << " \n";
            auto row = t->subMatches[1];
            // llvm::errs() << " row " << row << " \n";
            auto tileOp = rewriter.create<xilinx::AIE::TileOp>(
                tileCast.getLoc(), std::stoi(col), std::stoi(row));
            for (mlir::Operation *user : tileCast.getResult(0).getUsers()) {
              if (auto bufCast =
                      mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user)) {
                bufCast.emitRemark("Buffer cast from tile");
                auto mrt = bufferMemrefType(bufCast.getType(0), rewriter);
                auto bufferOp = rewriter.create<xilinx::AIE::BufferOp>(
                    bufCast.getLoc(), mrt, tileOp.getResult());
                // Keep track of the buffer op behind the C++ type
                cat->moduleTypes[bufCast.getType(0)]->data = bufferOp;
                // Erase the buffer cast which has been translated
                rewriter.eraseOp(bufCast);
                continue;
              }
              if (auto callOp = mlir::dyn_cast<mlir::cir::CallOp>(user)) {
                callOp.emitRemark("CallOp using a tile");
                if (isCallingFunctionWithAnnotation(callOp,
                                                    {"aie.tile.program"})) {
                  // if (false)
                  lowerTileProgram(tileOp, callOp, rewriter);
                  // Erase the scope owning the call operation representing the
                  // core op which has been translated
                  rewriter.eraseOp(
                      callOp->getParentOfType<mlir::cir::ScopeOp>());
                }
                continue;
              }
              user->emitError("User of tile cast not handled");
            }
          }
          // Erase the cast representing the tile op which has been translated
          rewriter.eraseOp(tileCast);
          continue;
        }
        user->emitRemark("User of device cast not handled");
      }
      // Replace the original cast by the same but connected to the device
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          op, op.getResult(0).getType(), deviceOp.getResult());
      for (mlir::Operation *user : op.getResult(0).getUsers()) {
        user->emitRemark("User of cast");
      }
      deviceOp->getParentOfType<mlir::cir::FuncOp>().emitRemark(
          "Rewritten function");
      return mlir::success();
    }
    return mlir::failure();
  }
};



struct CIRToAIE : CIRToAIEBase<CIRToAIE> {

  void deviceLowering(mlir::Operation *op, CIRToAIETypesAnalysis &cat) {
  llvm::SmallVector<mlir::Operation *> opsToErase;
  op->walk<mlir::WalkOrder::PreOrder>([&](mlir::UnrealizedConversionCastOp u) {
    u.emitRemark(
        "DeviceLowering found UnrealizedConversionCastOp inside the scope");
    if (!isUnrealizedConversionCastWithAnnotation(u, {"aie::device"}))
      return;
    if (auto aieLike = cat.moduleTypes[u.getType(0)];
        aieLike && aieLike->base == "aie::device") {
      auto deviceName = aieLike->subMatches[0];
      auto deviceId =
          xilinx::AIE::symbolizeEnum<xilinx::AIE::AIEDevice>(deviceName);
      if (!deviceId)
        // Actually this test cannot happens since the API of
        // xilinx::AIE::symbolizeEnum is strange: even if it returns a
        // std::optional it errors without returning
        u.emitError("aie::device incorrect for '") << deviceName << "'";
      // Create an aie.device just before its equivalent
      // UnrealizedConversionCast
      mlir::OpBuilder b{u};
      auto deviceOp = b.create<xilinx::AIE::DeviceOp>(u.getLoc(), *deviceId);
      // The aie.device requires one block
      deviceOp.getRegion().emplaceBlock();
      // Lazily move all the code depending on the device to the trash
      opsToErase.push_back(u);
    }
  });
  // Remove the useless operations
  eraseOpsAndUsers(opsToErase);
}

  void runOnOperation() override {
    // Compute the analysis for the module since it is a module pass.
    deviceLowering(getOperation(), getAnalysis<CIRToAIETypesAnalysis>());
#if 0
    // \todo Clean up this mess
    DeviceLoweringRP::cat = &cat;
    // See mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp
    mlir::ConversionTarget target{getContext()};
    target.addLegalDialect<xilinx::AIE::AIEDialect>();
    target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
        [](mlir::UnrealizedConversionCastOp op) {
          return !isUnrealizedConversionCastWithAnnotation(op, {"aie::device"});
        });
    mlir::RewritePatternSet patterns{&getContext()};
    patterns.add<DeviceLoweringRP>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
#endif
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCIRToAIEPreparePass() {
  return std::make_unique<CIRToAIEPrepare>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCIRToAIEPass() {
  return std::make_unique<CIRToAIE>();
}

} // namespace xilinx::AIE::CIR
