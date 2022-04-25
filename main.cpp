#include "desevi/mainwindow.h"

#include <QApplication>

#include "desevi/graph/MLIRModuleLoader.h"

#include "circt/Conversion/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

/// TODO: This could probably be done more elegantly by a modified TableGen
/// executable.
void initTransforms(TransformsRegistry &registry) {
  /*
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
*/
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

  TransformsRegistry registry;
  // initTransforms(registry);
  MainWindow w(registry);
  w.showMaximized();
  return a.exec();
}
