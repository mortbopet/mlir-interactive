#include "mlir-viewer/Scene.h"

#include "mlir-viewer/PassExecuter.h"
#include "mlir-viewer/graph/Edge.h"
#include "mlir-viewer/graph/NodeBase.h"
#include "mlir-viewer/graph/NodeSocket.h"

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

void Scene::executionFinished() {
  for (auto *item : items()) {
    auto *node = dynamic_cast<NodeBase *>(item);
    if (!node)
      continue;
    node->updateDrawState();
  }
}

Scene::Serialization Scene::getSerialization() const {
  Serialization s;
  std::map<NodeBase *, QString> nodeToID;
  std::map<QString, int> uniquer;

  // Create unique IDs for all nodes in the scene
  for (auto *item : itemsOfType<NodeBase>()) {
    nodeToID[item] =
        item->getName() + "_" + QString::number(uniquer[item->getName()]++);
  }

  // Serialize textual versions of the passes
  for (auto it : nodeToID)
    s.passes.push_back({it.second, it.first->getName()});

  // Create edge serializations from the NodeToID map.
  for (auto *edge : itemsOfType<Edge>()) {
    NodeBase *from = edge->getStartSocket()->getNode();
    int fromIdx = from->indexOfOutput(edge->getStartSocket());
    auto &fromID = nodeToID.at(from);

    NodeBase *to = edge->getEndSocket()->getNode();
    int toIdx = to->indexOfInput(edge->getEndSocket());
    auto &toID = nodeToID.at(to);

    assert(fromIdx >= 0 && toIdx >= 0);

    s.edges.push_back(Serialization::Edge{fromID, toID, fromIdx, toIdx});
  }
  return s;
}
