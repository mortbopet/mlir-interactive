#pragma once

#include "mlir-viewer/Support/FailureOr.h"
#include "mlir-viewer/Support/InflightResult.h"
#include "mlir-viewer/graph/BaseItem.h"
#include "mlir-viewer/graph/NodeSocket.h"

#include <QGraphicsRectItem>
#include <QGraphicsSimpleTextItem>
#include <QLayout>
#include <QObject>

#include <memory>
#include <string>
#include <vector>

using InflightNodeInputMapping = std::map<NodeSocket *, InflightResultBase *>;
struct ProcessInput {
  mlir::MLIRContext &context;
  InflightNodeInputMapping input;
};

template <typename TInner>
int indexOf(const std::vector<std::shared_ptr<TInner>> &container, TInner *v) {
  auto it =
      llvm::find_if(container, [&](const auto &it) { return it.get() == v; });
  if (it == container.end())
    return -1;
  return std::distance(container.begin(), it);
}

class NodeBase : public BaseGraphicsItem<QGraphicsRectItem> {
  Q_OBJECT

  template <typename TSocket>
  std::shared_ptr<NodeSocket>
  addSocket(const QString &name,
            std::vector<std::shared_ptr<NodeSocket>> &container,
            NodeType type) {
    auto socket = std::make_shared<TSocket>(name, type, this, this);
    container.push_back(socket);
    updateSockets();
    return socket;
  }

  void removeSocket(std::vector<std::shared_ptr<NodeSocket>> &container,
                    NodeSocket *socket) {
    auto it = std::find_if(container.begin(), container.end(),
                           [&](const auto &it) { return it.get() == socket; });
    assert(it != container.end() && "Socket not found");
    container.erase(it);
    updateSockets();
  }

public:
  NodeBase(const QString &name = "", QGraphicsItem *parent = nullptr);

  void contextMenuEvent(QGraphicsSceneContextMenuEvent *event) override;

  /// Returns the input sockets of the node.
  const std::vector<std::shared_ptr<NodeSocket>> &getInputs() const {
    return inputs;
  }
  /// Returns the output sockets of the node.
  const std::vector<std::shared_ptr<NodeSocket>> &getOutputs() const {
    return outputs;
  }

  int indexOfInput(NodeSocket *socket) const { return indexOf(inputs, socket); }
  int indexOfOutput(NodeSocket *socket) const {
    return indexOf(outputs, socket);
  }

  template <typename SocketType = NodeOutputSocket>
  SocketType *getOutput(int index) {
    assert(outputs.size() > index);
    return static_cast<SocketType *>(outputs.at(index).get());
  }

  template <typename SocketType = NodeInputSocket>
  SocketType *getInput(int index) {
    assert(inputs.size() > index);
    return static_cast<SocketType *>(outputs.at(index).get());
  }

  /// Add an input socket.
  std::shared_ptr<NodeSocket> addInput(const QString &name, NodeType type) {
    return addSocket<NodeInputSocket>(name, inputs, type);
  }

  /// Add an output socket.
  std::shared_ptr<NodeSocket> addOutput(const QString &name, NodeType type) {
    return addSocket<NodeOutputSocket>(name, outputs, type);
  }

  /// Remove an input socket.
  void removeInput(NodeSocket *socket) { removeSocket(inputs, socket); }

  /// Remove an output socket.
  void removeOutput(NodeSocket *socket) { removeSocket(outputs, socket); }

  void setName(const QString &name) override;

  /// Returns true if this is a source node. Source nodes define starting points
  /// in the compilation graph..
  virtual bool isSource() { return false; }

  /// Process an inflight result through this node.
  virtual ProcessResult process(ProcessInput input) = 0;

  void createUI(QVBoxLayout *layout) override;

  /// Updates the state of the item based on the IRState registered for the
  /// item.
  virtual void updateDrawState();

protected:
  /// Adjusts the position of input- and output sockets.
  void updateSockets();

private:
  QGraphicsPixmapItem *warningItem = nullptr;
  std::vector<std::shared_ptr<NodeSocket>> sockets;
  std::vector<std::shared_ptr<NodeSocket>> inputs;
  std::vector<std::shared_ptr<NodeSocket>> outputs;
  QGraphicsSimpleTextItem *textItem;
};

using NodeBuilder = std::function<NodeBase *()>;
