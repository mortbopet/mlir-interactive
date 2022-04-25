#include "desevi/graph/TransformNode.h"

#include "mlir/Pass/PassManager.h"

void TransformNode::addToPipeline(mlir::OpPassManager &pm) { nester(pm); }
