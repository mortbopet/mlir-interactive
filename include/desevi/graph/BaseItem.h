#pragma once

#include <QGraphicsItem>
#include <QLayout>

#include <memory>
#include <string>
#include <vector>

#include "desevi/Scene.h"

class BaseItem {
public:
  BaseItem(const QString &name) : name(name) {}
  virtual void setName(const QString &name) { this->name = name; }
  QString getName() const { return name; }

  /// Creates the user interface for this node on layout.
  virtual void createUI(QLayout *layout) {}

private:
  QString name;
};

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
};
