#include "desevi/graph/NodeSocket.h"
#include "desevi/graph/Edge.h"

#include <QBrush>
#include <QColor>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QPen>

NodeSocket::NodeSocket(const QString &name, NodeType type,
                       QGraphicsItem *parent)
    : QGraphicsEllipseItem(parent), type(type) {
  setFlag(QGraphicsItem::ItemSendsScenePositionChanges);
  setFlag(QGraphicsItem::ItemIsMovable, false);
  setFlag(QGraphicsItem::ItemIsSelectable, true);
  setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
  setAcceptHoverEvents(true);
  setAcceptDrops(false);
  setAcceptedMouseButtons(Qt::LeftButton);
  setZValue(1);
  setToolTip(name);
  setRect(-size / 2.0, -size / 2.0, size, size);

  setPen(QColor(Qt::black));
}

NodeInputSocket::NodeInputSocket(const QString &name, NodeType type,
                                 QGraphicsItem *parent)
    : NodeSocket(name, type, parent) {
  setBrush(Qt::red);
}

void NodeSocket::setEdge(std::shared_ptr<Edge> edge) {
  assert(!this->edge && "Socket already has an edge");
  this->edge = edge;
}

void NodeSocket::clearEdge() {
  assert(this->edge && "Socket has no edge");
  this->edge = nullptr;
}

QVariant NodeSocket::itemChange(QGraphicsItem::GraphicsItemChange change,
                                const QVariant &value) {
  if (change == QGraphicsItem::ItemScenePositionHasChanged) {
    if (edge) {
      edge->drawLineBetweenSockets();
    }
  }
  return QGraphicsEllipseItem::itemChange(change, value);
}

void NodeInputSocket::mousePressEvent(QGraphicsSceneMouseEvent *event) {
  if (!isConnected())
    return;

  // Forward the event to the output socket of the connected edge.
  auto edge = this->edge.get();
  auto outputSocket = edge->getStartSocket();
  outputSocket->clearEdge();
  clearEdge();
}

void NodeInputSocket::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
  // Do nothing; logic is contained in the output socket.
}

NodeOutputSocket::NodeOutputSocket(const QString &name, NodeType type,
                                   QGraphicsItem *parent)
    : NodeSocket(name, type, parent) {
  setBrush(Qt::green);
}

void NodeOutputSocket::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
  if (!connecting)
    return;

  // Update the edge's line based on the current mouse position.
  auto edge = this->edge.get();
  edge->drawLineTo(event->scenePos());
}

void NodeOutputSocket::mousePressEvent(QGraphicsSceneMouseEvent *event) {

  if (isConnected()) {
    QGraphicsItem::mousePressEvent(event);
    return;
  }

  // Create a new edge and add it to the scene
  auto edge = std::make_shared<Edge>(this);
  setEdge(edge);
  scene()->addItem(edge.get());
  connecting = true;

  QGraphicsItem::mousePressEvent(event);
}

void NodeOutputSocket::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
  if (!connecting)
    return;

  // Find any NodeInputSocket that is under the mouse cursor. If it is
  // unconnected and matches the type of this output socket, connect them.
  bool gotMatch = false;
  auto items = scene()->items(event->scenePos());
  for (auto &&item : items) {
    if (auto socket = dynamic_cast<NodeInputSocket *>(item)) {
      if (socket->isConnected())
        continue;

      if (socket->getType() == getType()) {
        // Found a match!
        socket->setEdge(edge);
        edge->setEndSocket(socket);
        gotMatch = true;
        break;
      }
      break;
    }
  }

  if (!gotMatch) {
    // No match found, so remove the edge
    auto edge = getEdge();
    clearEdge();
    scene()->removeItem(edge.get());
  }

  connecting = false;
}
