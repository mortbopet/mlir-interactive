#pragma once

#include <QString>
#include <functional>
#include <map>

class NodeBase;

namespace {
template <typename T>
NodeBase *defaultBuilder() {
  return new T();
}
} // namespace

using NodeBuilder = std::function<NodeBase *()>;

class TransformsRegistry {
public:
  void registerTransformation(const QString &name, NodeBuilder builder) {
    assert(transformations.find(name) == transformations.end() &&
           "Transformation already registered");
    transformations[name] = builder;
  }

  template <typename T>
  void registerTransformation(NodeBuilder builder = defaultBuilder<T>) {
    auto name = typeid(T).name();
    registerTransformation(name, builder);
  }

  const NodeBuilder &getBuilder(const QString &name) const {
    auto it = transformations.find(name);
    assert(it != transformations.end() && "Transformation not registered");
    return it->second;
  }

  const std::map<QString, NodeBuilder> &getTransformations() const {
    return transformations;
  }

private:
  std::map<QString, NodeBuilder> transformations;
};
