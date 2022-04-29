#include "mlir-interactive/graph/MLIRModuleLoader.h"
#include "mlir-interactive/NodeTypes.h"

#include "mlir/Parser/Parser.h"

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>

MLIRModuleLoader::MLIRModuleLoader(QGraphicsItem *parent)
    : NodeBase("MLIR module loader", parent) {

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

  NodeBase::createUI(layout);
}

void MLIRModuleLoader::setOutputType(const TypeKind &kind) {
  getOutputs().begin()->get()->setType(NodeType(kind));
}

QString MLIRModuleLoader::description() const {
  return "Loads an MLIR module from a file.";
}

ProcessResult MLIRModuleLoader::process(ProcessInput processInput) {
  NodeSocket *in = getInput(0);
  auto memBfr =
      dynamic_cast<InflightSource *>(processInput.input.at(getInput(0)).get());
  assert(memBfr && "Expected memory buffer as input!");
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBfr->getValue()), llvm::SMLoc());

  auto module = std::make_unique<mlir::OwningOpRef<mlir::ModuleOp>>();
  *module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &processInput.context);

  if (!module.get()->get())
    return processFailure() << "Failure during MLIR module loading!";

  return ResultMapping{
      {getOutput(0), std::make_shared<InflightModule>(std::move(module))}};
}
