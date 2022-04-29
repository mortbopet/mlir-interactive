#pragma once

#include <QString>
#include <map>
#include <optional>

#include <llvm/ADT/Any.h>

#include "mlir-interactive/IRState.h"
#include "mlir-interactive/graph/NodeBase.h"

class Scene;
class InflightResultBase;

namespace mlir {
class MLIRContext;
}

class PassExecuter {
public:
  PassExecuter(mlir::MLIRContext &context);

  /// Runs the pass executer on the nodes of the mlir-interactive scene.
  void execute(Scene &scene);
  std::optional<IRState> getState(void *item);

private:
  void executeNode(NodeBase *node,
                   std::map<NodeBase *, InflightNodeInputMapping> &nodeInputs);
  std::map<void *, IRState> IRStates;
  mlir::MLIRContext &context;

  // Handle for storing intercepted diagnostics information.
  QString diagnostic;
};
