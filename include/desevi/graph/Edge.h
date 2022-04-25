#pragma once

#include <QGraphicsLineItem>

#include "desevi/graph/NodeSocket.h"

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
  NodeOutputSocket *startSocket;
  NodeInputSocket *endSocket;
};
