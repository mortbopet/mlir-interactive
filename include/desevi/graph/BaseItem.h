#pragma once

#include <QGraphicsItem>
#include <QVBoxLayout>

#include <memory>
#include <string>
#include <vector>

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
