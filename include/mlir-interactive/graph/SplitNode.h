#pragma once

#include "mlir-interactive/graph/NodeBase.h"

/// A split pass takes a single MLIR module and emits N different outputs.

class SplitNode : public NodeBase {
  Q_OBJECT
public:
  SplitNode(QGraphicsItem *parent = nullptr);
  void createUI(QVBoxLayout *layout) override;
  QString description() const override;
  ProcessResult process(ProcessInput processInput) override;

private:
  void numOutputsChanged(int numOutputs);
};
