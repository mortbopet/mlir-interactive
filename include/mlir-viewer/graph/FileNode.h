#pragma once

#include "cereal/cereal.hpp"
#include "mlir-viewer/graph/NodeBase.h"

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

protected:
  QString filename;
};

class SourceFileNode : public FileNode {
public:
  SourceFileNode(const QString &filename, QGraphicsItem *parent = nullptr);
  QString description() const override;
  ProcessResult process(ProcessInput input) override;
  bool isSource() override { return true; }

private:
  std::shared_ptr<NodeSocket> outputSocket;
};

CEREAL_REGISTER_TYPE(FileNode);
