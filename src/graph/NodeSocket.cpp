#include "mlir-viewer/graph/NodeSocket.h"
#include "mlir-viewer/IRState.h"
#include "mlir-viewer/graph/Edge.h"

#include <QBrush>
#include <QColor>
#include <QGraphicsItemAnimation>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPen>
#include <QPlainTextEdit>
#include <QTimeLine>

NodeSocket::NodeSocket(const QString &name, NodeType type, NodeBase *node,
                       QGraphicsItem *parent)
    : BaseGraphicsItem<QGraphicsEllipseItem>(name, parent), type(type),
      node(node) {
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

  // Setup drop highlight
  dropHighlight = new QGraphicsEllipseItem(this);
  dropHighlight->setVisible(false);
  constexpr int penSize = 3;
  dropHighlight->setRect(rect().adjusted(-penSize, -penSize, penSize, penSize));
  auto pen = QPen(Qt::yellow, penSize);
  pen.setStyle(Qt::DotLine);
  dropHighlight->setPen(pen);
  dropHighlight->setZValue(0);

  dropHighlightTimer = new QTimeLine(2000);
  dropHighlightTimer->setLoopCount(0);
  dropHighlightTimer->setFrameRange(0, 100);
  dropHighlightTimer->setEasingCurve(QEasingCurve::Linear);

  dropHighlightAnimation = new QGraphicsItemAnimation;
  dropHighlightAnimation->setItem(dropHighlight);
  dropHighlightAnimation->setTimeLine(dropHighlightTimer);
  dropHighlightAnimation->setRotationAt(1.0, 360);
}

NodeSocket::~NodeSocket() {
  // Clear the edge before destroying this socket.
  if (edge) {
    edge->erase();
  }
}

void NodeSocket::enableDropHighlight(bool enabled) {
  dropHighlight->setVisible(enabled);
  if (enabled)
    dropHighlightTimer->start();
  else
    dropHighlightTimer->stop();
}

NodeInputSocket::NodeInputSocket(const QString &name, NodeType type,
                                 NodeBase *node, QGraphicsItem *parent)
    : NodeSocket(name, type, node, parent) {
  setBrush(Qt::red);
}

void NodeSocket::setEdge(std::shared_ptr<Edge> edge) {
  assert(!this->edge && "Socket already has an edge");
  this->edge = edge;
  connectionChanged();
}

void NodeSocket::clearEdge() {
  assert(this->edge && "Socket has no edge");
  this->edge = nullptr;
  connectionChanged();
}

template <typename TDerived, typename T1, typename T2>
bool isSameDerived(T1 a, T2 b) {
  return dynamic_cast<TDerived *>(a) && dynamic_cast<TDerived *>(b);
}

bool NodeSocket::isCompatible(NodeSocket *from) {
  // Same node?
  if (from->parentItem() == parentItem())
    return false;

  // Same direction?
  if (isSameDerived<NodeInputSocket>(this, from) ||
      isSameDerived<NodeOutputSocket>(this, from))
    return false;

  // Already connected?
  if (hasEdge())
    return false;

  return getType().isCompatible(from->getType());
}

void NodeSocket::setType(NodeType type) {
  if (edge) {
    // Clear the edge in case the new type does is incompatible.
    NodeSocket *otherSocket =
        edge->getEndSocket() == this
            ? static_cast<NodeSocket *>(edge->getStartSocket())
            : edge->getEndSocket();

    if (!otherSocket->getType().isCompatible(type)) {
      edge->erase();
      nodeChanged();
    }
  }

  this->type = type;
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

  auto IRState = static_cast<Scene *>(scene())->getIRStateForItem(this);
  if (IRState.has_value()) {
    auto IRTextEdit = new QPlainTextEdit();
    IRTextEdit->setPlainText(IRState.value().getIR());
    layout->addWidget(IRTextEdit);
  }
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
                                   NodeBase *node, QGraphicsItem *parent)
    : NodeSocket(name, type, node, parent) {
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
  static_cast<Scene *>(scene())->highlightCompatibleSockets(this, getType());

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

      if (socket->isCompatible(this)) {
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
  static_cast<Scene *>(scene())->clearSocketHighlight();
  return NodeSocket::mouseReleaseEvent(event);
}
