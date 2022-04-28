#pragma once

#include "desevi/Scene.h"
#include "desevi/graph/BaseItem.h"

template <typename T = QGraphicsItem>
class BaseGraphicsItem : public BaseItem, public T {
public:
  BaseGraphicsItem(const QString &name = "", QGraphicsItem *parent = nullptr)
      : BaseItem(name), T(parent) {
    T::setFlag(QGraphicsItem::ItemIsSelectable);
  }

  QVariant itemChange(QGraphicsItem::GraphicsItemChange change,
                      const QVariant &value) override {
    if (change == QGraphicsItem::ItemSelectedChange && value == true) {
      static_cast<Scene *>(T::scene())->requestFocus(this);
    }

    return T::itemChange(change, value);
  }

  void nodeChanged() { static_cast<Scene *>(T::scene())->graphChanged(); }
};
