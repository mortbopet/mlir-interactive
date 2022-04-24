#include "desevi/graph/FileNode.h"

#include <QFileInfo>

void FileNode::setFilename(const QString &filename) {
  this->filename = filename;
}

SourceFileNode::SourceFileNode(const QString &filename, QGraphicsItem *parent)
    : FileNode("", parent) {
  setFilename(filename);

  // Get the base filename from the path.
  QString baseFilename = QFileInfo(filename).fileName();
  setNodeName(baseFilename);

  // Add an output socket representing the file.
  outputSocket = addOutput("output", NodeType::File);
}
