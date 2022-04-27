#pragma once

#include <QGraphicsScene>
#include <cereal/archives/json.hpp>

#include "desevi/NodeTypes.h"

class BaseItem;
class NodeSocket;
class PassExecuter;
class IRState;

class Scene : public QGraphicsScene {
  Q_OBJECT
public:
  Scene(PassExecuter &executer, QObject *parent = nullptr);

  void requestFocus(BaseItem *item);

  void highlightCompatibleSockets(NodeSocket *source, NodeType sourceType);
  void clearSocketHighlight();

  /// Returns any registered in-flight information for the selected pipeline
  /// item.
  std::optional<IRState> getIRStateForItem(BaseItem *item);

  /// Called by items in the scene whenever a change to compilation graph
  /// occurs.
  void graphChanged();

  /// Called after a pass execution cycle has finished, and the UI should be
  /// updated to reflect the resulting state.
  void executionFinished();

signals:
  void focusItem(BaseItem *);

  /*
  template <class Archive>
  void store(const QString &filepath) const {
    // Open file for writing
    std::ofstream os(filepath.toStdString());
    Archive oarchive oa(os);

    // Serealize the scene
    oarchive(*this);
  }

  template <class Archive>
  void load(const QString &filepath) {
    archive(cereal::make_nvp("nodes", nodes));
  }

  template <class Archive>
  void serialize(Archive &ar) {
    // Serialize nodes, then edges
    for (auto item : items()) {
      if (auto node = dynamic_cast<NodeBase *>(item)) {
        ar(cereal::make_nvp("node", *node));
      }
    }

    for (auto item : items()) {
      if (auto node = dynamic_cast<NodeBase *>(item)) {
        ar(cereal::make_nvp("node", *node));
      }
    }
  }
*/

private:
  std::vector<NodeSocket *> highlightedSockets;
  PassExecuter &executer;
};
