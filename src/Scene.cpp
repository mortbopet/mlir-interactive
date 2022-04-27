#include "desevi/Scene.h"

#include "desevi/PassExecuter.h"
#include "desevi/graph/NodeBase.h"
#include "desevi/graph/NodeSocket.h"

Scene::Scene(PassExecuter &executer, QObject *parent)
    : QGraphicsScene(parent), executer(executer) {}

void Scene::requestFocus(BaseItem *node) { focusItem(node); }

void Scene::highlightCompatibleSockets(NodeSocket *source,
                                       NodeType sourceType) {
  for (auto *item : items()) {
    if (auto socket = dynamic_cast<NodeSocket *>(item)) {
      if (socket->isCompatible(source)) {
        socket->enableDropHighlight(true);
        highlightedSockets.push_back(socket);
      }
    }
  }
}

void Scene::clearSocketHighlight() {
  for (auto socket : highlightedSockets) {
    socket->enableDropHighlight(false);
  }
  highlightedSockets.clear();
}

std::optional<IRState> Scene::getIRStateForItem(BaseItem *item) {
  return executer.getState(item);
}

void Scene::graphChanged() { executer.execute(*this); }
