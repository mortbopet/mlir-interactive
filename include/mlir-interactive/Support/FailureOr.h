#pragma once

#include "mlir/Support/LogicalResult.h"

namespace mv {

/// Like mlir::FailureOr but additionally carries an error message.
template <typename T>
class FailureOr : public mlir::FailureOr<T> {
public:
  using mlir::FailureOr<T>::FailureOr;
  const std::string &getError() const { return m_errorMessage; }
  FailureOr<T> operator<<(const std::string &other) {
    m_errorMessage += other;
    return *this;
  }

private:
  std::string m_errorMessage;
};

template <typename T>
inline FailureOr<T> success(bool isSuccess = true) {
  return FailureOr<T>(mlir::LogicalResult::success(isSuccess));
}

/// Utility function to generate a LogicalResult. If isFailure is true a
/// `failure` result is generated, otherwise a 'success' result is generated.
template <typename T>
inline FailureOr<T> failure(bool isFailure = true) {
  return FailureOr<T>(mlir::LogicalResult::failure(isFailure));
}

} // namespace mlirviewer
