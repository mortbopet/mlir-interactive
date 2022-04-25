#include "desevi/graph/MLIRModuleLoader.h"
#include "desevi/NodeTypes.h"

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>

MLIRModuleLoader::MLIRModuleLoader(QGraphicsItem *parent)
    : NodeBase("MLIR Module", parent) {

  addInput("input file", NodeType(TypeKind::AnyFile));
  addOutput("MLIR module", NodeType(TypeKind::AnyMLIR));
}

void MLIRModuleLoader::createUI(QVBoxLayout *layout) {
  // Create a combobox with the available MLIR types.
  auto *mlirTypeComboBox = new QComboBox();
  for (auto t : mlirKinds()) {
    mlirTypeComboBox->addItem(t._to_string());
  }

  // Also allow AnyMLIR
  TypeKind anyMLIR = TypeKind::AnyMLIR;
  mlirTypeComboBox->addItem(anyMLIR._to_string());

  auto *hlayout = new QHBoxLayout();
  hlayout->addWidget(new QLabel("MLIR type:"));
  hlayout->addWidget(mlirTypeComboBox);
  layout->addLayout(hlayout);
}
