#pragma once

#include "desevi/graph/NodeBase.h"

/// A node which loads an input file into an MLIR module. The MLIR module will
/// post-loading be represented as a typed IR node.

class MLIRModuleLoader : public NodeBase {
public:
  MLIRModuleLoader(QGraphicsItem *parent = nullptr);

  void setOutputType(const QString &type);

  void createUI(QVBoxLayout *layout) override;

private:
  QString outputType;
};
