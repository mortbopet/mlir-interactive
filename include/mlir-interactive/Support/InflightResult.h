#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <QString>

#include <variant>

#include "mlir-interactive/Support/FailureOr.h"

class NodeOutputSocket;

class InflightResultBase {
public:
  virtual ~InflightResultBase(){};

  virtual QString toString() = 0;
};

template <typename T>
class InflightResult : public InflightResultBase {
public:
  InflightResult(T value) : value(std::move(value)) {}
  InflightResult(InflightResult &other) : value(std::move(other.value)) {}
  T &getValue() { return value; }

private:
  T value;
};

using EmptyInflightResult = InflightResult<std::monostate>;

class InflightModule : public InflightResult<
                           std::unique_ptr<mlir::OwningOpRef<mlir::ModuleOp>>> {
public:
  using InflightResult::InflightResult;
  QString toString() override;
};

class InflightSource
    : public InflightResult<std::unique_ptr<llvm::MemoryBuffer>> {
public:
  using InflightResult::InflightResult;
  QString toString() override;
};

/// A ResultMapping defines an output path for a process result. Pass execution
/// will continue through the denoted NodeOutputSockets using the provided
/// result.
using ResultMapping =
    std::map<NodeOutputSocket *, std::shared_ptr<InflightResultBase>>;
using ProcessResult = mv::FailureOr<ResultMapping>;

/// Utility function to generate a LogicalResult. If isFailure is true a
/// `failure` result is generated, otherwise a 'success' result is generated.
inline mv::FailureOr<ResultMapping> processFailure(bool isFailure = true) {
  return mv::FailureOr<ResultMapping>(mlir::LogicalResult::failure(isFailure));
}
