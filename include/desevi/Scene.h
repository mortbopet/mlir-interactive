#pragma once

#include <QGraphicsScene>
#include <cereal/archives/json.hpp>

class BaseItem;

class Scene : public QGraphicsScene {
  Q_OBJECT
public:
  Scene(QObject *parent = nullptr);

  void requestFocus(BaseItem *node);

signals:
  void focusNode(BaseItem *);

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
};
