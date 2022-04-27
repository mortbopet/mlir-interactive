#pragma once

#include "desevi/FailureOr.h"
#include "desevi/InflightResult.h"
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

  template <class Archive>
  void serialize(Archive &ar) {
    auto name = getName();
    ar(name);
    setName(name);
  }

  void setName(const QString &name) override;

  /// Returns true if this is a source node. Source nodes define starting points
  /// in the compilation graph..
  virtual bool isSource() { return false; }

  /// Process an inflight result through this node.
  virtual ProcessResult process(ProcessInput input) = 0;

  void createUI(QVBoxLayout *layout) override;

protected:
  /// Adjusts the position of input- and output sockets.
  void updateSockets();

private:
  std::vector<std::shared_ptr<NodeSocket>> sockets;
  std::vector<std::shared_ptr<NodeSocket>> inputs;
  std::vector<std::shared_ptr<NodeSocket>> outputs;
  QGraphicsSimpleTextItem *textItem;
};

using NodeBuilder = std::function<NodeBase *()>;
