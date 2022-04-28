
#include "mlir-viewer/graph/NodeBase.h"

/// An IR node represents an in-memory IR representation of some MLIR module.
/// The IR node is a typed representation of the MLIR module.
class IRNode : public NodeBase {
public:
  IRNode(const QString &name = "", QGraphicsItem *parent = nullptr);
};