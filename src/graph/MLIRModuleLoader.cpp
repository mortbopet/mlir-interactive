

#include "desevi/graph/MLIRModuleLoader.h"

MLIRModuleLoader::MLIRModuleLoader(QGraphicsItem *parent)
    : NodeBase("MLIR Module", parent) {

  addInput("input file", NodeType(TypeKind::AnyFile));
  addOutput("MLIR module", NodeType(TypeKind::AnyMLIR));
}
