#pragma once

#include "desevi/graph/NodeBase.h"

#include <QString>

/// A pipeline node is able to contain other transformation nodes, in a reusable
/// and serializeable manner.
class PipelineNode : public NodeBase {
public:
  TransformNode(QGraphicsItem *parent = nullptr);

  /// Return the input type of this pipeline. The pipeline inherits the input
  /// type of whatever transformation it connects to internally.
  NodeType inputType() const;

  /// Returns the output type of this pipeline. The pipeline inherits the input
  /// type of whatever transformation it connects to internally.
  NodeType outputType() const;
};
