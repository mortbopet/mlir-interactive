#pragma once

#include "desevi/graph/NodeBase.h"

#include <QString>
#include <functional>
#include <memory>

namespace mlir {
class OpPassManager;
class Pass;
} // namespace mlir

/// A function which adds a pass to a pass manager.
using PassManagerNester = std::function<void(mlir::OpPassManager &)>;

/// A TransformNode is able to ingest an IRNode and transform it into some other
/// IR node.
class TransformNode : public NodeBase {
public:
  using NodeBase::NodeBase;

  void setInputType(NodeType type) { inputType = type; }
  void setOutputType(NodeType type) { outputType = type; }

  static NodeBuilder getBuilder(NodeType inputType, NodeType outputType,
                                const QString &name,
                                const PassManagerNester &nester) {
    return [=]() -> NodeBase * {
      TransformNode *node = new TransformNode(name);
      node->addInput("input", inputType);
      node->addOutput("output", outputType);
      node->nester = nester;
      return node;
    };
  }

  void addToPipeline(mlir::OpPassManager &pm);
  ProcessResult process(ProcessInput processInput) override;

private:
  NodeType inputType = NodeType(TypeKind::None);
  NodeType outputType = NodeType(TypeKind::None);
  PassManagerNester nester;
};
