#include "desevi/PassExecuter.h"
#include "desevi/Scene.h"
#include "desevi/graph/Edge.h"
#include "desevi/graph/NodeBase.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include <QStringList>

using namespace mlir;

PassExecuter::PassExecuter(mlir::MLIRContext &context) : context(context) {
  // Install a diagnostics handler to intersect info/error output.
  context.getDiagEngine().registerHandler([this](mlir::Diagnostic &diag) {
    if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
      std::string err;
      llvm::raw_string_ostream os(err);
      os << diag.getLocation() << ": " << diag;
      this->diagnostic = QString::fromStdString(err);
      return success();
    }
    return failure();
  });
}

void PassExecuter::executeNode(NodeBase *node, InflightResultBase *input) {
  auto result = node->process(ProcessInput{context, input});
  if (failed(result)) {
    QString errorMessage = QString::fromStdString(result.getError());
    errorMessage += "\n" + diagnostic;
    IRStates[node] = IRState(errorMessage, /*isError=*/true);
    diagnostic.clear();
  } else {
    // Gather IR state and continue execution through output sockets
    for (auto &&[outputSocket, outputRes] : result.getValue()) {
      IRStates[outputSocket] = outputRes->toString();

      if (!outputSocket->hasEdge())
        continue;

      NodeSocket *toSocket = outputSocket->getEdge()->getEndSocket();
      executeNode(toSocket->getNode(), outputRes.get());
    }
  }
}

void PassExecuter::execute(Scene &scene) {
  // Gather source nodes.
  std::vector<NodeBase *> sourceNodes;
  IRStates.clear();

  for (auto *item : scene.items()) {
    auto *node = dynamic_cast<NodeBase *>(item);
    if (!node)
      continue;
    if (!node->isSource())
      continue;
    sourceNodes.push_back(node);
  };

  // Go execute!
  for (auto *sourceNode : sourceNodes)
    executeNode(sourceNode, nullptr);

  scene.executionFinished();
}

std::optional<IRState> PassExecuter::getState(void *item) {
  auto it = IRStates.find(item);
  if (it == IRStates.end())
    return std::nullopt;
  return it->second;
}
