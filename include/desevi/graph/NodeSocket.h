#pragma once

#include <QGraphicsEllipseItem>

#include <memory>

#include "desevi/NodeTypes.h"
#include "desevi/graph/BaseItem.h"

QT_BEGIN_NAMESPACE
class QTimeLine;
class QGraphicsItemAnimation;
QT_END_NAMESPACE

class Edge;
class NodeBase;

class NodeSocket : public BaseGraphicsItem<QGraphicsEllipseItem> {
  Q_OBJECT
  constexpr static double size = 15.0;

public:
  NodeSocket(const QString &name, NodeType type, NodeBase *node,
             QGraphicsItem *parent);
  ~NodeSocket() override;

  QVariant itemChange(QGraphicsItem::GraphicsItemChange change,
                      const QVariant &value) override;

  bool hasEdge() { return static_cast<bool>(edge); }
  const std::shared_ptr<Edge> &getEdge() { return edge; }
  void setEdge(std::shared_ptr<Edge> edge);
  void clearEdge();
  bool isConnected() const { return static_cast<bool>(edge); }

  /// Node type compatability check which also disallows self-connections and
  /// in-in/out-out connections.
  bool isCompatible(NodeSocket *socket);
  NodeType getType() const { return type; }
  void setType(NodeType type);
  NodeBase *getNode() { return node; }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(type);
  }

  void createUI(QVBoxLayout *layout) override;

  /// Called by the scene whenever an edge is being created and this socket is a
  /// valid drop target.
  void enableDropHighlight(bool enabled);

signals:
  void connectionChanged();

protected:
  std::shared_ptr<Edge> edge;
  NodeType type;

private:
  QGraphicsEllipseItem *dropHighlight = nullptr;
  QGraphicsItemAnimation *dropHighlightAnimation = nullptr;
  QTimeLine *dropHighlightTimer = nullptr;
  NodeBase *node = nullptr;
};

class NodeInputSocket : public NodeSocket {

public:
  NodeInputSocket(const QString &name, NodeType type, NodeBase *node,
                  QGraphicsItem *parent);
  void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
};

class NodeOutputSocket : public NodeSocket {
  friend class NodeInputSocket;

public:
  NodeOutputSocket(const QString &name, NodeType type, NodeBase *node,
                   QGraphicsItem *parent);

protected:
  void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
  void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
  void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;

private:
  bool connecting = false;
};
