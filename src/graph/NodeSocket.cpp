#include "desevi/graph/NodeSocket.h"
#include "desevi/graph/Edge.h"

#include <QBrush>
#include <QColor>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPen>

NodeSocket::NodeSocket(const QString &name, NodeType type,
                       QGraphicsItem *parent)
    : BaseGraphicsItem<QGraphicsEllipseItem>(name, parent), type(type) {
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

NodeSocket::~NodeSocket() {
  // Clear the edge before destroying this socket.
  if (edge)
    edge->clear();
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
  return BaseGraphicsItem<QGraphicsEllipseItem>::itemChange(change, value);
}

void NodeSocket::createUI(QVBoxLayout *layout) {
  // Create a label and line edit showing the type of the socket.
  auto *hlayout = new QHBoxLayout();
  auto typeLabel = new QLineEdit(type.toString());
  typeLabel->setReadOnly(true);
  hlayout->addWidget(new QLabel("Type:"));
  hlayout->addWidget(typeLabel);
  layout->addLayout(hlayout);
}

void NodeInputSocket::mousePressEvent(QGraphicsSceneMouseEvent *event) {
  if (!isConnected())
    return NodeSocket::mousePressEvent(event);

  // Forward the event to the output socket of the connected edge.
  auto edge = this->edge.get();
  auto outputSocket = edge->getStartSocket();
  outputSocket->clearEdge();
  clearEdge();
  return NodeSocket::mousePressEvent(event);
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
    NodeSocket::mousePressEvent(event);
    return;
  }

  // Create a new edge and add it to the scene
  auto edge = std::make_shared<Edge>(this);
  setEdge(edge);
  scene()->addItem(edge.get());
  connecting = true;

  NodeSocket::mousePressEvent(event);
}

void NodeOutputSocket::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
  if (!connecting)
    return NodeSocket::mouseReleaseEvent(event);

  // Find any NodeInputSocket that is under the mouse cursor. If it is
  // unconnected and matches the type of this output socket, connect them.
  bool gotMatch = false;
  auto items = scene()->items(event->scenePos());
  for (auto &&item : items) {
    if (auto socket = dynamic_cast<NodeInputSocket *>(item)) {
      if (socket->isConnected())
        continue;

      if (socket->getType().isCompatible(getType())) {
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
  return NodeSocket::mouseReleaseEvent(event);
}
