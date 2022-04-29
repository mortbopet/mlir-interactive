#pragma once

#include "mlir-interactive/graph/NodeBase.h"

/// A merge pass takes N different inputs of AnyMLIR and emit a single
/// MLIR module, where all modules have been merged into one.

class MergeNode : public NodeBase {
  Q_OBJECT
public:
  MergeNode(QGraphicsItem *parent = nullptr);
  void createUI(QVBoxLayout *layout) override;
  QString description() const override;
  ProcessResult process(ProcessInput processInput) override;

private:
  void numInputsChanged(int numInputs);
};
