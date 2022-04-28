#include "desevi/mainwindow.h"

#include <QApplication>

#include "desevi/graph/MLIRModuleLoader.h"

#include "circt/Conversion/Passes.h"
#include "circt/InitAllDialects.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/InitLLVM.h"

/// TODO: This could probably be done more elegantly by a modified TableGen
/// executable. That TableGen executable should also add additional reflection
/// about the options of passes. Currently, this is nested deeply within type
/// erased, private, and .cpp containing logic, which is impossible to even hack
/// our way through to inspect.
void initTransforms(TransformsRegistry &registry) {
  registry.registerTransformation<MLIRModuleLoader>();
  registry.registerTransformation(
      "Passthrough", NodeType(TypeKind::AnyMLIR), NodeType(TypeKind::AnyMLIR),
      [](mlir::OpPassManager &pm) { assert(false && "how do we do this?"); });
  registry.registerTransformation(
      "mlir-clang", NodeType({TypeKind::C, TypeKind::CPP}),
      NodeType(TypeKind::AnyMLIR),
      [](mlir::OpPassManager &pm) { assert(false && "how do we do this?"); });
  registry.registerTransformation("Affine lowering", NodeType(TypeKind::Affine),
                                  NodeType(TypeKind::Standard),
                                  [](mlir::OpPassManager &pm) {
                                    pm.addPass(mlir::createLowerAffinePass());
                                  });
  registry.registerTransformation(
      "Standard to Handshake", NodeType(TypeKind::Standard),
      NodeType(TypeKind::Handshake), [](mlir::OpPassManager &pm) {
        pm.addPass(circt::createStandardToHandshakePass());
      });
  registry.registerTransformation(
      "Handshake to FIRRTL", NodeType(TypeKind::Handshake),
      NodeType(TypeKind::FIRRTL), [](mlir::OpPassManager &pm) {
        pm.addPass(circt::createHandshakeToFIRRTLPass());
      });
}

int main(int argc, char *argv[]) {
  llvm::InitLLVM y(argc, argv);
  QApplication a(argc, argv);

  mlir::DialectRegistry dialectRegistry;
  dialectRegistry.insert<mlir::AffineDialect>();
  dialectRegistry.insert<mlir::LLVM::LLVMDialect>();
  dialectRegistry.insert<mlir::memref::MemRefDialect>();
  dialectRegistry.insert<mlir::func::FuncDialect>();
  dialectRegistry.insert<mlir::arith::ArithmeticDialect>();
  dialectRegistry.insert<mlir::cf::ControlFlowDialect>();
  dialectRegistry.insert<mlir::scf::SCFDialect>();
  circt::registerAllDialects(dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.allowUnregisteredDialects();

  TransformsRegistry registry;
  initTransforms(registry);
  MainWindow w(context, registry);
  w.showMaximized();
  return a.exec();
}
