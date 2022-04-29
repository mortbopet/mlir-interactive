#include "mlir-interactive/Support/InflightResult.h"

QString InflightSource::toString() {
  return QString::fromStdString(getValue().get()->getBuffer().str());
}

QString InflightModule::toString() {
  std::string module;
  llvm::raw_string_ostream os(module);
  getValue().get()->get().print(os);
  return QString::fromStdString(module);
}
