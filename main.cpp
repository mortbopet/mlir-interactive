#include "desevi/mainwindow.h"

#include <QApplication>

#include "desevi/graph/MLIRModuleLoader.h"

void initTransforms(TransformsRegistry &registry) {
  registry.registerTransformation<MLIRModuleLoader>();
  registry.registerTransformation("Passthrough", NodeType(TypeKind::AnyMLIR),
                                  NodeType(TypeKind::AnyMLIR));
  registry.registerTransformation("mlir-clang",
                                  NodeType({TypeKind::C, TypeKind::CPP}),
                                  NodeType(TypeKind::AnyMLIR));
  registry.registerTransformation("Affine to Standard",
                                  NodeType(TypeKind::Affine),
                                  NodeType(TypeKind::Standard));
  registry.registerTransformation("Standard to Handshake",
                                  NodeType(TypeKind::Standard),
                                  NodeType(TypeKind::Handshake));
  registry.registerTransformation("Handshake to FIRRTL",
                                  NodeType(TypeKind::Handshake),
                                  NodeType(TypeKind::FIRRTL));
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

  TransformsRegistry registry;
  initTransforms(registry);
  MainWindow w(registry);
  w.showMaximized();
  return a.exec();
}
