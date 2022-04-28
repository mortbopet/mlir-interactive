#pragma once

#include "desevi/Support/FailureOr.h"
#include "desevi/Support/InflightResult.h"
#include "desevi/graph/BaseItem.h"
#include "desevi/graph/NodeSocket.h"

#include <QGraphicsRectItem>
#include <QGraphicsSimpleTextItem>
#include <QLayout>
#include <QObject>

#include <memory>
#include <string>
#include <vector>

struct ProcessInput {
  mlir::MLIRContext &context;
  InflightResultBase *input;
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
  template <typename... Args>
  std::shared_ptr<NodeSocket> addInput(const QString &name, NodeType type,
                                       Args &&...args) {
    auto socket = std::make_shared<NodeInputSocket>(
        name, type, this, this, std::forward<Args>(args)...);
    inputs.push_back(socket);
    updateSockets();
    return socket;
  }

  /// Add an output socket.
  template <typename... Args>
  std::shared_ptr<NodeSocket> addOutput(const QString &name, NodeType type,
                                        Args &&...args) {
    auto socket = std::make_shared<NodeOutputSocket>(
        name, type, this, this, std::forward<Args>(args)...);
    outputs.push_back(socket);
    updateSockets();
    return socket;
  }

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
