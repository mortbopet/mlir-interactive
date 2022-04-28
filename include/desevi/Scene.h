#pragma once

#include <QGraphicsScene>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

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

  struct Serialization {
    struct Node {
      QString passID;   // Unique ID of the instance
      QString passName; // Name corresponding to a registered pass.

      template <class Archive>
      void serialize(Archive &ar) {
        ar(CEREAL_NVP(passID));
        ar(CEREAL_NVP(passName));
      }
    };

    // A list of mappings of [passID : passName]
    std::vector<Node> passes;

    struct Edge {
      QString fromPassID;
      QString toPassID;
      int fromPassOutput;
      int toPassInput;
      template <class Archive>
      void serialize(Archive &ar) {
        ar(CEREAL_NVP(fromPassID));
        ar(CEREAL_NVP(toPassID));
        ar(CEREAL_NVP(fromPassOutput));
        ar(CEREAL_NVP(toPassInput));
      }
    };
    std::vector<Edge> edges;

    template <class Archive>
    void serialize(Archive &ar) {
      ar(passes, edges);
    }
  };

  template <class Archive>
  void save(Archive &ar) const {
    ar(getSerialization());
  }

  template <class Archive>
  void load(Archive &ar) {}

  template <typename T>
  std::vector<T *> itemsOfType() const {
    std::vector<T *> vs;
    for (auto *item : items()) {
      auto *v = dynamic_cast<T *>(item);
      if (!v)
        continue;
      vs.push_back(v);
    }
    return vs;
  }

signals:
  void focusItem(BaseItem *);

private:
  Serialization getSerialization() const;

  std::vector<NodeSocket *> highlightedSockets;
  PassExecuter &executer;
};

template <class Archive>
void serialize(Archive &archive, QString &m) {
  std::string s = m.toStdString();
  archive(s);
  m = QString::fromStdString(s);
}
