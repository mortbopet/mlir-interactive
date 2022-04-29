#include "mlir-viewer/graph/Edge.h"

#include <QPen>

Edge::Edge(NodeOutputSocket *startSocket)
    : QGraphicsLineItem(nullptr), startSocket(startSocket), endSocket(nullptr) {
  setPen(QPen(Qt::black, 2));
  setZValue(0);
}

void Edge::setEndSocket(NodeInputSocket *endSocket) {
  // It is a precondition that the endsocket has already been registered as an
  // owner of this socket.
  if (endSocket)
    assert(endSocket->getEdge().get() == this);
  this->endSocket = endSocket;
  drawLineBetweenSockets();
  edgeChanged();
}

void Edge::drawLineBetweenSockets() {
  assert(startSocket && endSocket);
  setLine(startSocket->scenePos().x(), startSocket->scenePos().y(),
          endSocket->scenePos().x(), endSocket->scenePos().y());
}

void Edge::drawLineTo(QPointF pos) {
  setLine(startSocket->scenePos().x(), startSocket->scenePos().y(), pos.x(),
          pos.y());
}

void Edge::erase() {
  startSocket->clearEdge();
  if (endSocket)
    endSocket->clearEdge();
  startSocket = nullptr;
  endSocket = nullptr;
}

void Edge::edgeChanged() { static_cast<Scene *>(scene())->graphChanged(); }
