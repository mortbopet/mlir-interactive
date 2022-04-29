#include "mlir-viewer/PassExecuter.h"
#include "mlir-viewer/Scene.h"
#include "mlir-viewer/graph/Edge.h"
#include "mlir-viewer/graph/NodeBase.h"

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

ProcessResult PassExecuter::executeNode(
    NodeBase *node,
    std::map<NodeBase *, InflightNodeInputMapping> &nodeInputs) {
  auto result = node->process(ProcessInput{context, nodeInputs[node]});
  if (failed(result)) {
    QString errorMessage = QString::fromStdString(result.getError());
    errorMessage += "\n" + diagnostic;
    IRStates[node] = IRState(errorMessage, /*isError=*/true);
    diagnostic.clear();
  } else {
    // Report IR state on output sockets and prepare inputs for connected output
    // nodes.
    for (auto &&[outputSocket, outputRes] : result.getValue()) {
      IRStates[outputSocket] = outputRes->toString();
      auto *endSocket = outputSocket->getEdge()->getEndSocket();
      nodeInputs[endSocket->getNode()][endSocket] = outputRes.get();
    }
  }
  return result;
}

void PassExecuter::execute(Scene &scene) {
  // Topologically sort the graph.
  // @todo: this obviously assumes a DAG, which might not always be the case!
  std::vector<NodeBase *> sortedNodes = scene.getNodesSorted();
  std::map<NodeBase *, InflightNodeInputMapping> nodeInputs;

  // Go execute!
  for (auto *node : sortedNodes) {
    auto result = executeNode(node, nodeInputs);
    if (failed(result))
      break;

    // Drop the node input buffer since it's no longer needed.
    nodeInputs.erase(node);
  }

  scene.executionFinished();
}

std::optional<IRState> PassExecuter::getState(void *item) {
  auto it = IRStates.find(item);
  if (it == IRStates.end())
    return std::nullopt;
  return it->second;
}
