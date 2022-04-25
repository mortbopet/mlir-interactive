#include "desevi/Scene.h"

Scene::Scene(QObject *parent) : QGraphicsScene(parent) {}

void Scene::requestFocus(BaseItem *node) { emit focusItem(node); }
