#pragma once

#include <QString>
#include <map>
#include <optional>

#include <llvm/ADT/Any.h>

#include "desevi/IRState.h"

class Scene;
class NodeBase;
class InflightResultBase;

namespace mlir {
class MLIRContext;
}

class PassExecuter {
public:
  PassExecuter(mlir::MLIRContext &context);

  /// Runs the pass executer on the nodes of the Desevi scene.
  void execute(Scene &scene);
  std::optional<IRState> getState(void *item);

private:
  void executeNode(NodeBase *node, InflightResultBase *input);
  std::map<void *, IRState> IRStates;
  mlir::MLIRContext &context;

  // Handle for storing intercepted diagnostics information.
  QString diagnostic;
};
