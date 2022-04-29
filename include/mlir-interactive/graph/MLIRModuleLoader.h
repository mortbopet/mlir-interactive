#pragma once

#include "mlir-interactive/graph/NodeBase.h"

/// A node which loads an input file into an MLIR module. The MLIR module will
/// post-loading be represented as a typed IR node.

class MLIRModuleLoader : public NodeBase {
  Q_OBJECT
public:
  MLIRModuleLoader(QGraphicsItem *parent = nullptr);
  void setOutputType(const TypeKind &type);
  void createUI(QVBoxLayout *layout) override;
  QString description() const override;

  ProcessResult process(ProcessInput processInput) override;
};
