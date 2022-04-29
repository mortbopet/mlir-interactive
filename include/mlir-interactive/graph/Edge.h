#pragma once

#include <QGraphicsLineItem>

#include "mlir-interactive/graph/NodeSocket.h"

class Edge : public QGraphicsLineItem {
public:
  Edge(NodeOutputSocket *startSocket);

  void setEndSocket(NodeInputSocket *endSocket);
  NodeOutputSocket *getStartSocket() const { return startSocket; }
  NodeInputSocket *getEndSocket() const { return endSocket; }
  void erase();

  void drawLineBetweenSockets();
  void drawLineTo(QPointF pos);

private:
  /// Called by this object on construction, destruction and end socket change
  /// to notify the scene that graph connectivity was modified.
  void edgeChanged();

  NodeOutputSocket *startSocket;
  NodeInputSocket *endSocket;
};
