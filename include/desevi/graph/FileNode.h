#pragma once

#include "desevi/graph/NodeBase.h"

#include <QString>

class FileNode : public NodeBase {
public:
  using NodeBase::NodeBase;
  void setFilename(const QString &filename);

private:
  QString filename;
};

class SourceFileNode : public FileNode {
public:
  SourceFileNode(const QString &filename, QGraphicsItem *parent = nullptr);

private:
  std::shared_ptr<NodeSocket> outputSocket;
};
