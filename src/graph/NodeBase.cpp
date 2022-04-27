#include "desevi/graph/NodeBase.h"
#include "desevi/IRState.h"
#include "desevi/Scene.h"

#include "cereal/cereal.hpp"

#include <QBrush>
#include <QGraphicsPixmapItem>
#include <QGraphicsSceneContextMenuEvent>
#include <QMenu>
#include <QPlainTextEdit>

NodeBase::NodeBase(const QString &name, QGraphicsItem *parent)
    : BaseGraphicsItem<QGraphicsRectItem>(name, parent) {
  setFlag(ItemIsMovable);
  setCacheMode(DeviceCoordinateCache);
  setZValue(-1);
  setBrush(Qt::white);

  textItem = new QGraphicsSimpleTextItem(this);
  setName(name);
}

void NodeBase::setName(const QString &name) {
  BaseItem::setName(name);
  textItem->setText(name);
  auto br = textItem->boundingRect();
  br.moveTo(-br.width() / 2, -br.height() / 2);
  textItem->setPos(QPointF(-br.width() / 2, -br.height() / 2));
  setRect(br.adjusted(-br.height() / 3, -br.height() / 2, br.height() / 2,
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

void NodeBase::contextMenuEvent(QGraphicsSceneContextMenuEvent *event) {
  auto menu = new QMenu();
  menu->addAction("Delete", [this]() {
    Scene *_scene = static_cast<Scene *>(scene());
    _scene->removeItem(this);
    delete this;
    _scene->graphChanged();
  });
  menu->exec(event->screenPos());
}

void NodeBase::createUI(QVBoxLayout *layout) {
  // Report any error informatiomn related to this pass.
  auto state = static_cast<Scene *>(scene())->getIRStateForItem(this);
  if (state.has_value() && state->isError()) {
    auto errorTextEdit = new QPlainTextEdit();
    errorTextEdit->setPlainText(state.value().getError());
    layout->addWidget(errorTextEdit);
  }

  BaseItem::createUI(layout);
}

void NodeBase::updateDrawState() {
  auto irState = static_cast<Scene *>(scene())->getIRStateForItem(this);

  if (!irState.has_value() || !irState->isError()) {
    if (warningItem)
      delete warningItem;
  } else {
    // Scale the warningItem based on the font size (which scales to the DPI of
    // the screen).
    QSizeF sizeGuide = QSizeF(QFontMetrics(textItem->font())
                                  .boundingRect(QStringLiteral("!"))
                                  .size()) *
                       1.5;
    warningItem = new QGraphicsPixmapItem(
        QIcon(":/icons/warning.svg")
            .pixmap(sizeGuide.height(), sizeGuide.height()),
        this);
    int d = warningItem->boundingRect().height() / 2;
    warningItem->setPos(rect().topLeft() - QPointF(d, d));
  }
}
