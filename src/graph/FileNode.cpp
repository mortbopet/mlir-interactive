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

TypeKind FileNode::inferKindFromExtension(const QString &filename) {
  QString extension = QFileInfo(filename).suffix();
  if (extension == "cpp")
    return TypeKind::CPP;
  else if (extension == "c")
    return TypeKind::C;
  else
    return TypeKind::None;
}

void FileNode::createUI(QVBoxLayout *layout) {
  auto *textViewer = new QPlainTextEdit();
  textViewer->setReadOnly(true);
  QFile file(filename);
  if (file.open(QIODevice::ReadOnly)) {
    QTextStream in(&file);
    textViewer->setPlainText(in.readAll());
  }
  textViewer->setSizePolicy(QSizePolicy::MinimumExpanding,
                            QSizePolicy::MinimumExpanding);
  layout->addWidget(textViewer);
}

SourceFileNode::SourceFileNode(const QString &filename, QGraphicsItem *parent)
    : FileNode("", parent) {
  // Add an output socket representing the file.
  outputSocket =
      addOutput("output", NodeType(inferKindFromExtension(filename)));
  setFilename(filename);
}
