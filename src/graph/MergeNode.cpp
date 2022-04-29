#include "mlir-viewer/graph/MergeNode.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>

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
  return {

  };
}
