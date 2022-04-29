#include "mlir-viewer/graph/MergeNode.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>

#include "mlir/IR/Builders.h"

using namespace mlir;

MergeNode::MergeNode(QGraphicsItem *parent) : NodeBase("Merge", parent) {
  numInputsChanged(1);
  addOutput("Output", NodeType(TypeKind::AnyMLIR));
}

void MergeNode::numInputsChanged(int numInputs) {
  int currentNumInputs = getInputs().size();
  assert(std::abs(currentNumInputs - numInputs) <= 1);
  if (numInputs < currentNumInputs)
    removeInput(&*getInputs().back());
  else if (numInputs > currentNumInputs)
    addInput("Input " + QString::number(currentNumInputs + 1),
             NodeType(TypeKind::AnyMLIR));
}

void MergeNode::createUI(QVBoxLayout *layout) {
  QHBoxLayout *hbox = new QHBoxLayout();
  hbox->addWidget(new QLabel("# inputs: "));

  // todo: box should probably change with # of inputs/outputs. This change
  // should be made in NodeBase.
  QSpinBox *numInputs = new QSpinBox();
  numInputs->setMinimum(1);
  numInputs->setMaximum(10);
  numInputs->setValue(getInputs().size());
  connect(numInputs, qOverload<int>(&QSpinBox::valueChanged), this,
          &MergeNode::numInputsChanged);
  hbox->addWidget(numInputs);

  layout->addLayout(hbox);
}

QString MergeNode::description() const { return "Merge"; }

ProcessResult MergeNode::process(ProcessInput processInput) {

  auto iterateInputModules = [&](const std::function<void(ModuleOp)> &f) {
    for (auto &inputSocket : getInputs()) {
      auto inflightModule = dynamic_cast<InflightModule *>(
          processInput.input.at(inputSocket.get()).get());
      assert(inflightModule && "expected module input");
      ModuleOp inputModule = inflightModule->getValue()->get();
      f(inputModule);
    }
  };

  // Build a fused location from all source modules.
  SmallVector<Location> locations;
  iterateInputModules(
      [&](ModuleOp inputModule) { locations.push_back(inputModule.getLoc()); });
  auto loc = FusedLoc::get(&processInput.context, locations);

  // Build the new module contain all merged modules.
  OpBuilder builder(&processInput.context);
  auto mergedModule = builder.create<mlir::ModuleOp>(loc);
  auto mergedModuleRef =
      std::make_unique<mlir::OwningOpRef<mlir::ModuleOp>>(mergedModule);

  // Inline all of the input modules.
  iterateInputModules([&](ModuleOp inputModule) {
    Block *source = inputModule.getBody();
    Block *dest = mergedModule.getBody();
    dest->getOperations().splice(dest->end(), source->getOperations());
  });

  return ResultMapping{{getOutput(0), std::make_shared<InflightModule>(
                                          std::move(mergedModuleRef))}};
}
