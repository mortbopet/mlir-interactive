#include "desevi/mainwindow.h"
#include "ui_mainwindow.h"

#include <QAction>
#include <QGraphicsView>
#include <QListView>
#include <QStandardItemModel>
#include <QToolBar>

#include "desevi/graph/MLIRModuleLoader.h"
#include "desevi/graph/TransformNode.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);

  scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  ui->graphicsView->setScene(scene);

  // Mock file explorer
  auto *fileModel = new QStandardItemModel(this);
  ui->explorer->setModel(fileModel);
  auto *dummyItem = new QStandardItem("dummyfile.cpp");
  dummyItem->setFlags(dummyItem->flags() & ~Qt::ItemIsEditable);
  fileModel->setItem(0, 0, dummyItem);

  connect(ui->explorer, &QTreeView::doubleClicked, this,
          &MainWindow::explorerDoubleClicked);

  setupTransforms();
  setupActions();
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::explorerDoubleClicked(const QModelIndex &index) {
  QString filename = index.data().toString();
  // Create a new source file node.
  auto *node = new SourceFileNode(filename, nullptr);
  scene->addItem(node);
}

void MainWindow::setupActions() {
  // Create a toolbar in the main window.
  auto *toolbar = new QToolBar("Toolbar", this);
  addToolBar(toolbar);

  interactiveAction = new QAction("Interactive", this);
  interactiveAction->setCheckable(true);
  interactiveAction->setChecked(false);
  toolbar->addAction(interactiveAction);

  runAction = new QAction("Run", this);
  toolbar->addAction(runAction);
}

void MainWindow::setupTransforms() {
  auto *transformsModel = new QStandardItemModel(this);
  ui->transforms->setModel(transformsModel);
  connect(ui->transforms, &QListView::doubleClicked, this,
          &MainWindow::transformsDoubleClicked);

  registry.registerTransformation<MLIRModuleLoader>();
  registry.registerTransformation(
      "Passthrough",
      TransformNode::getBuilder(NodeType::Any, NodeType::Any, "Passthrough"));

  for (auto it : registry.getTransformations()) {
    transformsModel->appendRow(new QStandardItem(it.first));
  }
}

void MainWindow::transformsDoubleClicked(const QModelIndex &index) {
  QString transformName = index.data().toString();
  // Create a new transform node.
  auto *node = registry.getBuilder(index.data().toString())();
  scene->addItem(node);
}
