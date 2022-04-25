#pragma once

#include <QGraphicsEllipseItem>

#include <memory>

#include "desevi/NodeTypes.h"
#include "desevi/graph/BaseItem.h"

class Edge;

class NodeSocket : public BaseGraphicsItem<QGraphicsEllipseItem> {
  constexpr static double size = 15.0;

public:
  NodeSocket(const QString &name, NodeType type,
             QGraphicsItem *parent = nullptr);

  QVariant itemChange(QGraphicsItem::GraphicsItemChange change,
                      const QVariant &value);

  const std::shared_ptr<Edge> &getEdge() { return edge; }
  void setEdge(std::shared_ptr<Edge> edge);
  void clearEdge();
  bool isConnected() const { return static_cast<bool>(edge); }
  NodeType getType() const { return type; }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(type);
  }

protected:
  std::shared_ptr<Edge> edge;
  NodeType type;
};

class NodeInputSocket : public NodeSocket {

public:
  NodeInputSocket(const QString &name, NodeType type,
                  QGraphicsItem *parent = nullptr);
  void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
  void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
};

class NodeOutputSocket : public NodeSocket {
  friend class NodeInputSocket;

public:
  NodeOutputSocket(const QString &name, NodeType type,
                   QGraphicsItem *parent = nullptr);

protected:
  void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
  void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
  void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;

private:
  bool connecting = false;
};
