#include "desevi/graph/FileNode.h"

#include <QFileInfo>
#include <QPlainTextEdit>
#include <QTextStream>

void FileNode::setFilename(const QString &filename) {
  this->filename = filename;

  // Get the base filename from the path.
  QString baseFilename = QFileInfo(filename).fileName();
  setName(baseFilename);
}

void FileNode::createUI(QLayout *layout) {
  auto *textViewer = new QPlainTextEdit();
  textViewer->setReadOnly(true);
  QFile file(filename);
  if (file.open(QIODevice::ReadOnly)) {
    QTextStream in(&file);
    textViewer->setPlainText(in.readAll());
  }
  layout->addWidget(textViewer);
}

SourceFileNode::SourceFileNode(const QString &filename, QGraphicsItem *parent)
    : FileNode("", parent) {
  // Add an output socket representing the file.
  outputSocket = addOutput("output", NodeType::File);
  setFilename(filename);
}
