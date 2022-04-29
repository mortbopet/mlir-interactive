#include "mlir-interactive/graph/PassNode.h"

#include "mlir/Pass/PassManager.h"

using namespace mlir;

void PassNode::addToPipeline(mlir::OpPassManager &pm) { nester(pm); }

ProcessResult PassNode::process(ProcessInput processInput) {
  auto inputModule =
      dynamic_cast<InflightModule *>(processInput.input.at(getInput(0)).get());
  assert(inputModule && "expected module input");
  ModuleOp module = inputModule->getValue()->get();

  PassManager pm(&processInput.context);
  nester(pm);

  if (failed(pm.run(module)))
    return processFailure() << "Error during pass execution";

  return ResultMapping{{getOutput(0), std::make_shared<InflightModule>(
                                          std::move(inputModule->getValue()))}};
}
