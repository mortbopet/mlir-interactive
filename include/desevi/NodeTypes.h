#pragma once

#include "better-enums/enum.h"
#include <QString>
#include <algorithm>
#include <assert.h>
#include <set>
#include <vector>

BETTER_ENUM(TypeKind, int,
            // Special cases
            None, Any,
            // File kinds
            AnyFile, C, CPP,
            // MLIR kinds
            AnyMLIR, Affine, Handshake);

/// Not a huge fan of these functions - what's needed is a form of
/// nested/hierarchical enums, which we don't have in C++. With this, things
/// may be buggy if we forget to update the functions...
static inline std::set<TypeKind> fileKinds() {
  return {TypeKind::C, TypeKind::CPP};
}
static inline std::set<TypeKind> mlirKinds() {
  return {TypeKind::Affine, TypeKind::Handshake};
}

/// Kinds which are not allowed to be mixed with any other kind.
static inline std::set<TypeKind> unaryKinds() {
  return {TypeKind::Any, TypeKind::None, TypeKind::AnyMLIR, TypeKind::AnyFile};
}

class NodeType {
public:
  NodeType(const std::set<TypeKind> &kinds) { setKinds(kinds); }
  NodeType(TypeKind kind) { setKind(kind); }
  const std::set<TypeKind> &getKinds() const { return kinds; }
  void setKinds(const std::set<TypeKind> &kinds);
  void setKind(TypeKind kind) { setKinds({kind}); }
  void addKind(TypeKind kind);
  bool isCompatible(const NodeType &other) const;

  bool operator==(const NodeType &other) const { return kinds == other.kinds; }

  QString toString() const;

private:
  bool isUnaryKind() const {
    return kinds.size() == 1 && unaryKinds().count(*kinds.begin());
  }

  TypeKind getUnaryKind() const {
    assert(isUnaryKind());
    return *kinds.begin();
  }

  std::set<TypeKind> kinds;
};
