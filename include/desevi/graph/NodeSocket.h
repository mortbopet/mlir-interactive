#pragma once

#include <QGraphicsEllipseItem>

#include <memory>

#include "desevi/NodeTypes.h"
#include "desevi/graph/BaseItem.h"

class Edge;
class QTimeLine;
class QGraphicsItemAnimation;

class NodeSocket : public BaseGraphicsItem<QGraphicsEllipseItem> {
  constexpr static double size = 15.0;

public:
  NodeSocket(const QString &name, NodeType type,
             QGraphicsItem *parent = nullptr);
  ~NodeSocket() override;

  QVariant itemChange(QGraphicsItem::GraphicsItemChange change,
                      const QVariant &value) override;

  const std::shared_ptr<Edge> &getEdge() { return edge; }
  void setEdge(std::shared_ptr<Edge> edge);
  void clearEdge();
  bool isConnected() const { return static_cast<bool>(edge); }

  /// Node type compatability check which also disallows self-connections and
  /// in-in/out-out connections.
  bool isCompatible(NodeSocket *socket);
  NodeType getType() const { return type; }
  void setType(NodeType type);

  template <class Archive>
  void serialize(Archive &ar) {
    ar(type);
  }

  void createUI(QVBoxLayout *layout) override;

  /// Called by the scene whenever an edge is being created and this socket is a
  /// valid drop target.
  void enableDropHighlight(bool enabled);

protected:
  std::shared_ptr<Edge> edge;
  NodeType type;

private:
  QGraphicsEllipseItem *dropHighlight = nullptr;
  QGraphicsItemAnimation *dropHighlightAnimation = nullptr;
  QTimeLine *dropHighlightTimer = nullptr;
};

class NodeInputSocket : public NodeSocket {

public:
  NodeInputSocket(const QString &name, NodeType type,
                  QGraphicsItem *parent = nullptr);
  void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
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
