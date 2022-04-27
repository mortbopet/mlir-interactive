#pragma once

#include <QGraphicsItem>
#include <QVBoxLayout>

#include <memory>
#include <string>
#include <vector>

#include "desevi/Scene.h"

class BaseItem : public QObject {
  Q_OBJECT
public:
  BaseItem(const QString &name) : name(name) {}
  virtual ~BaseItem(){};
  virtual void setName(const QString &name) { this->name = name; }
  QString getName() const { return name; }

  /// Creates the user interface for this node on layout.
  virtual void createUI(QVBoxLayout *layout) {}
  virtual QString description() const { return QString(); }

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

  void nodeChanged() { static_cast<Scene *>(T::scene())->graphChanged(); }
};
