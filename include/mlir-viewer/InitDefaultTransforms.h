#include "mlir-viewer/TransformsRegistry.h"

#include "mlir-viewer/graph/MLIRModuleLoader.h"
#include "mlir-viewer/graph/MergeNode.h"
#include "mlir-viewer/graph/SplitNode.h"

static inline void initDefaultTransforms(TransformsRegistry &registry) {
  registry.registerTransformation<MLIRModuleLoader>();
  registry.registerTransformation(
      "Passthrough", NodeType(TypeKind::AnyMLIR), NodeType(TypeKind::AnyMLIR),
      [](mlir::OpPassManager &pm) { assert(false && "how do we do this?"); });
  registry.registerTransformation<MergeNode>();
  registry.registerTransformation<SplitNode>();
}
