

#include "desevi/graph/MLIRModuleLoader.h"

MLIRModuleLoader::MLIRModuleLoader(QGraphicsItem *parent)
    : NodeBase("MLIR Module", parent) {

  addInput("input file", NodeType::File);
  addOutput("MLIR module", NodeType::Unset);
}
