#include "mlir-interactive/PassExecuter.h"
#include "mlir-interactive/Scene.h"
#include "mlir-interactive/graph/Edge.h"
#include "mlir-interactive/graph/NodeBase.h"

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

void PassExecuter::executeNode(
    NodeBase *node,
    std::map<NodeBase *, InflightNodeInputMapping> &nodeInputs) {
  auto result = node->process(ProcessInput{context, nodeInputs[node]});
  if (failed(result)) {
    QString errorMessage = QString::fromStdString(result.getError());
    errorMessage += "\n" + diagnostic;
    IRStates[node] = IRState(errorMessage, /*isError=*/true);
    diagnostic.clear();
  } else {
    for (auto &&[outputSocket, outputRes] : result.getValue()) {
      // Report IR state on output sockets
      IRStates[outputSocket] = outputRes->toString();
      if (!outputSocket->hasEdge())
        continue;

      // prepare inputs for connected output nodes.
      NodeSocket *endSocket = outputSocket->getEdge()->getEndSocket();
      NodeBase *endNode = endSocket->getNode();
      nodeInputs[endNode][endSocket] = outputRes;

      // If a connected node has all inputs available, execute it.
      if (llvm::all_of(endNode->getInputs(), [&](auto &inputSocket) {
            return nodeInputs[endNode].count(inputSocket.get());
          }))
        executeNode(endNode, nodeInputs);
    }
  }

  // Drop the node input buffer since it's no longer needed.
  nodeInputs.erase(node);
}

void PassExecuter::execute(Scene &scene) {
  // Topologically sort the graph.
  // @todo: this obviously assumes a DAG, which might not always be the case!
  std::vector<NodeBase *> sortedNodes =
      scene.getNodesSorted(/*skipUnconnected=*/true);
  std::map<NodeBase *, InflightNodeInputMapping> nodeInputs;

  // Go execute!
  for (auto *node : scene.itemsOfType<NodeBase>()) {
    if (node->isSource())
      executeNode(node, nodeInputs);
  }

  scene.executionFinished();
}

std::optional<IRState> PassExecuter::getState(void *item) {
  auto it = IRStates.find(item);
  if (it == IRStates.end())
    return std::nullopt;
  return it->second;
}
