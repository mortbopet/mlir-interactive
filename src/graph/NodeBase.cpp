#include "desevi/graph/NodeBase.h"

#include <QBrush>

NodeBase::NodeBase(const QString &name, QGraphicsItem *parent)
    : QGraphicsRectItem(parent) {

  setFlag(ItemIsSelectable);
  setFlag(ItemIsMovable);
  setFlag(ItemSendsGeometryChanges);
  setCacheMode(DeviceCoordinateCache);
  setZValue(-1);

  textItem = new QGraphicsSimpleTextItem(this);
  setNodeName(name);
}

void NodeBase::setNodeName(const QString &name) {
  textItem->setText(name);
  auto br = textItem->boundingRect();
  br.moveTo(-br.width() / 2, -br.height() / 2);
  textItem->setPos(QPointF(-br.width() / 3, -br.height() / 2));
  setRect(br.adjusted(-br.width() / 3, -br.height() / 2, br.width() / 2,
                      br.height() / 2));
  updateSockets();
}

void NodeBase::updateSockets() {
  auto rect = boundingRect();
  auto topDiff = rect.width() / (inputs.size() + 1);
  auto botDiff = rect.width() / (outputs.size() + 1);
  auto y = rect.height() / 2;

  int indent = -rect.width() / 2 + topDiff;
  for (auto &&socket : inputs) {
    socket->setPos(indent, -y);
    indent += topDiff;
  }
  indent = -rect.width() / 2 + botDiff;
  for (auto &&socket : outputs) {
    socket->setPos(0, y);
    indent += botDiff;
  }
}
