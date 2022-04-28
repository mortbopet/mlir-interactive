#include "mlir-viewer/NodeTypes.h"

#include <QStringList>

bool NodeType::isCompatible(const NodeType &other) const {
  assert(other.getKinds().size() == 1 &&
         "Expected matching kind set size to be unary");
  TypeKind target = *other.getKinds().begin();

  // Most general case; None or Any
  if (other.kinds.count(TypeKind::None))
    return false;

  if (kinds.count(TypeKind::Any))
    return true;

  // Subcases; any MLIR file or any source file
  if (isUnaryKind()) {
    if (getUnaryKind() == target)
      return true;

    switch (*kinds.begin()) {
    case TypeKind::AnyMLIR:
      return mlirKinds().count(target);
    case TypeKind::AnyFile:
      return fileKinds().count(target);
    default:
      assert(false && "Unhandled unary kind!");
    }
  }

  // Fallback case; check if other->kinds is a subset of this->kinds.
  return std::includes(kinds.begin(), kinds.end(), other.kinds.begin(),
                       other.kinds.end());
}

void NodeType::setKinds(const std::set<TypeKind> &kinds) {
  for (auto k : kinds) {
    if (unaryKinds().count(k) && kinds.size() > 1)
      assert(false && "Unary kind within multiple kinds!");
  }

  this->kinds = kinds;
}

void NodeType::addKind(TypeKind kind) {
  auto kindsCopy = kinds;
  kindsCopy.insert(kind);
  setKinds(kindsCopy);
}

QString NodeType::toString() const {
  QStringList strs;
  for (auto k : kinds) {
    strs.append(QString::fromStdString(k._to_string()));
  }
  return strs.join(", ");
}
