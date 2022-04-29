#include "mlir-interactive/graph/FileNode.h"

#include <QFileInfo>
#include <QPlainTextEdit>
#include <QTextStream>

#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace mlir;

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
    return TypeKind::AnyFile;
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

QString SourceFileNode::description() const { return "Loads a file."; }

ProcessResult SourceFileNode::process(ProcessInput processInput) {
  assert(processInput.input.size() == 0 && "Input to a source node?");
  std::string errorMessage;
  auto input = mlir::openInputFile(filename.toStdString(), &errorMessage);
  if (!input) {
    return processFailure() << errorMessage << "\n";
  }
  return ResultMapping{
      {getOutput(0), std::make_shared<InflightSource>(std::move(input))}};
}
