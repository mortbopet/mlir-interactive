#pragma once

#include "desevi/graph/NodeBase.h"

#include "cereal/cereal.hpp"

#include <cereal/types/polymorphic.hpp>

#include <QString>

class FileNode : public NodeBase {
public:
  using NodeBase::NodeBase;
  void setFilename(const QString &filename);

  void createUI(QVBoxLayout *layout) override;

  template <class Archive>
  void serialize(Archive &ar) {
    // ar(cereal::base_class<NodeBase>(this), filename);
  }

  static TypeKind inferKindFromExtension(const QString &filename);

private:
  QString filename;
};

class SourceFileNode : public FileNode {
public:
  SourceFileNode(const QString &filename, QGraphicsItem *parent = nullptr);

private:
  std::shared_ptr<NodeSocket> outputSocket;
};

CEREAL_REGISTER_TYPE(FileNode);
