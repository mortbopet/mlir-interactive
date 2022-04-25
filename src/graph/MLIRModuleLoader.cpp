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
  connect(mlirTypeComboBox, &QComboBox::currentTextChanged, this,
          [this](const QString &text) {
            this->setOutputType(
                TypeKind::_from_string(text.toStdString().c_str()));
          });

  // Also allow AnyMLIR
  TypeKind anyMLIR = TypeKind::AnyMLIR;
  mlirTypeComboBox->addItem(anyMLIR._to_string());

  mlirTypeComboBox->setCurrentText(
      getOutputs().begin()->get()->getType().getSingleKind()._to_string());

  auto *hlayout = new QHBoxLayout();
  hlayout->addWidget(new QLabel("MLIR type:"));
  hlayout->addWidget(mlirTypeComboBox);
  layout->addLayout(hlayout);
}

void MLIRModuleLoader::setOutputType(const TypeKind &kind) {
  getOutputs().begin()->get()->setType(NodeType(kind));
}

QString MLIRModuleLoader::description() const {
  return "Loads an MLIR module from a file.";
}
