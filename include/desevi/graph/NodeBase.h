#pragma once

#include "desevi/graph/NodeSocket.h"

#include <QGraphicsRectItem>
#include <QGraphicsSimpleTextItem>

#include <memory>
#include <string>
#include <vector>

class NodeBase : public QGraphicsRectItem {
public:
  NodeBase(const QString &name = "", QGraphicsItem *parent = nullptr);

  /// Returns the input sockets of the node.
  const std::vector<std::shared_ptr<NodeSocket>> &getInputs() const {
    return inputs;
  }
  /// Returns the output sockets of the node.
  const std::vector<std::shared_ptr<NodeSocket>> &getOutputs() const {
    return outputs;
  }

  /// Add an input socket.
  template <typename... Args>
  std::shared_ptr<NodeSocket> addInput(const QString &name, NodeType type,
                                       Args &&...args) {
    auto socket = std::make_shared<NodeInputSocket>(
        name, type, this, std::forward<Args>(args)...);
    inputs.push_back(socket);
    updateSockets();
    return socket;
  }

  /// Add an output socket.
  template <typename... Args>
  std::shared_ptr<NodeSocket> addOutput(const QString &name, NodeType type,
                                        Args &&...args) {
    auto socket = std::make_shared<NodeOutputSocket>(
        name, type, this, std::forward<Args>(args)...);
    outputs.push_back(socket);
    updateSockets();
    return socket;
  }

  /// Set the node name.
  void setNodeName(const QString &name);

protected:
  /// Adjusts the position of input- and output sockets.
  void updateSockets();

private:
  std::vector<std::shared_ptr<NodeSocket>> sockets;
  std::vector<std::shared_ptr<NodeSocket>> inputs;
  std::vector<std::shared_ptr<NodeSocket>> outputs;
  QGraphicsSimpleTextItem *textItem;
};
