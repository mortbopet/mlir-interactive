#include "desevi/graph/TransformNode.h"

#include "mlir/Pass/PassManager.h"

using namespace mlir;

void TransformNode::addToPipeline(mlir::OpPassManager &pm) { nester(pm); }

ProcessResult TransformNode::process(ProcessInput processInput) {
  auto inputModule = dynamic_cast<InflightModule *>(processInput.input);
  assert(inputModule && "expected module input");
  ModuleOp module = inputModule->getValue()->get();

  PassManager pm(&processInput.context);
  nester(pm);

  if (failed(pm.run(module)))
    return processFailure() << "Error during pass execution";

  return ResultMapping{{getOutput(0), std::make_shared<InflightModule>(
                                          std::move(inputModule->getValue()))}};
}
