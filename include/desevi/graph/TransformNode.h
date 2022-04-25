#pragma once

#include "desevi/TransformsRegistry.h"
#include "desevi/graph/NodeBase.h"

#include <QString>

/// A TransformNode is able to ingest an IRNode and transform it into some other
/// IR node.
class TransformNode : public NodeBase {
public:
  using NodeBase::NodeBase;

  void setInputType(NodeType type) { inputType = type; }
  void setOutputType(NodeType type) { outputType = type; }

  static NodeBuilder getBuilder(NodeType inputType, NodeType outputType,
                                const QString &name) {
    return [=]() -> NodeBase * {
      TransformNode *node = new TransformNode(name);
      node->addInput("input", inputType);
      node->addOutput("output", outputType);
      return node;
    };
  }

private:
  NodeType inputType = NodeType(TypeKind::None);
  NodeType outputType = NodeType(TypeKind::None);
};
