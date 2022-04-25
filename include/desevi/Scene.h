#pragma once

#include <QGraphicsScene>
#include <cereal/archives/json.hpp>

#include "desevi/NodeTypes.h"

class BaseItem;
class NodeSocket;

class Scene : public QGraphicsScene {
  Q_OBJECT
public:
  Scene(QObject *parent = nullptr);

  void requestFocus(BaseItem *item);

  void highlightCompatibleSockets(NodeSocket *source, NodeType sourceType);
  void clearSocketHighlight();

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
};
