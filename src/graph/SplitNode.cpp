#include "mlir-viewer/graph/SplitNode.h"

#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QSpinBox>
#include <QTableWidget>

SplitNode::SplitNode(QGraphicsItem *parent) : NodeBase("Split", parent) {
  addInput("Input", NodeType(TypeKind::AnyMLIR));
  numOutputsChanged(1);
}

void SplitNode::numOutputsChanged(int numOutputs) {
  int currentNumOutputs = getOutputs().size();
  assert(std::abs(currentNumOutputs - numOutputs) <= 1);
  if (numOutputs < currentNumOutputs)
    removeOutput(&*getOutputs().back());
  else if (numOutputs > currentNumOutputs)
    addOutput("Output " + QString::number(currentNumOutputs + 1),
              NodeType(TypeKind::AnyMLIR));
}

void SplitNode::createUI(QVBoxLayout *layout) {
  QHBoxLayout *hbox = new QHBoxLayout();
  hbox->addWidget(new QLabel("# outputs: "));

  QSpinBox *numOutputs = new QSpinBox();
  numOutputs->setMinimum(1);
  numOutputs->setMaximum(10);
  numOutputs->setValue(getOutputs().size());
  connect(numOutputs, qOverload<int>(&QSpinBox::valueChanged), this,
          &SplitNode::numOutputsChanged);
  hbox->addWidget(numOutputs);
  layout->addLayout(hbox);

  QTableWidget *table = new QTableWidget();
  table->setColumnCount(2);
  table->setHorizontalHeaderLabels({"Symbol", "Output #"});
  table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);

  layout->addWidget(table);
}

QString SplitNode::description() const { return "Split"; }

ProcessResult SplitNode::process(ProcessInput processInput) { return {}; }