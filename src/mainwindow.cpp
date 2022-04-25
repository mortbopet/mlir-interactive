#include "desevi/mainwindow.h"
#include "ui_mainwindow.h"

#include <QAction>
#include <QDirIterator>
#include <QFileDialog>
#include <QGraphicsView>
#include <QListView>
#include <QSpacerItem>
#include <QStandardItemModel>
#include <QToolBar>
#include <QVBoxLayout>

#include "desevi/Scene.h"
#include "desevi/graph/MLIRModuleLoader.h"
#include "desevi/graph/TransformNode.h"

void clearLayout(QLayout *layout) {
  // Clear focuslayout
  QLayoutItem *layoutItem;
  while ((layoutItem = layout->takeAt(0)) != nullptr) {
    if (auto *subLayout = dynamic_cast<QLayout *>(layoutItem))
      clearLayout(subLayout);
    delete layoutItem->widget();
    delete layoutItem;
  }
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);

  scene = new Scene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  ui->graphicsView->setScene(scene);

  // Mock file explorer
  fileModel = new QStandardItemModel(this);
  ui->explorer->setModel(fileModel);

  connect(ui->explorer, &QTreeView::doubleClicked, this,
          &MainWindow::explorerDoubleClicked);

  setupTransforms();
  setupActions();

  // Connect item focus changes to updates in the focus UI layout.
  connect(scene, &Scene::focusItem, this, [&](BaseItem *item) {
    clearLayout(ui->focusLayout);

    auto subLayout = new QVBoxLayout();
    ui->focusLayout->addLayout(subLayout);

    if (item)
      item->createUI(subLayout);

    // subLayout->addSpacing(1);
    ui->focusLayout->addSpacerItem(new QSpacerItem(
        1, 1, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding));
  });
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

  openFolderAction = new QAction("Open Folder", this);
  ui->menuFile->addAction(openFolderAction);
  connect(openFolderAction, &QAction::triggered, this,
          &MainWindow::openFolderClicked);
}

void MainWindow::setupTransforms() {
  auto *transformsModel = new QStandardItemModel(this);
  ui->transforms->setModel(transformsModel);
  connect(ui->transforms, &QListView::doubleClicked, this,
          &MainWindow::transformsDoubleClicked);

  registry.registerTransformation<MLIRModuleLoader>();
  registry.registerTransformation(
      "Passthrough",
      TransformNode::getBuilder(NodeType(TypeKind::AnyMLIR),
                                NodeType(TypeKind::AnyMLIR), "Passthrough"));

  registry.registerTransformation(
      "mlir-clang",
      TransformNode::getBuilder(NodeType({TypeKind::C, TypeKind::CPP}),
                                NodeType(TypeKind::AnyMLIR), "mlir-clang"));

  for (auto it : registry.getTransformations()) {
    auto *tx = new QStandardItem(it.first);
    transformsModel->appendRow(tx);
    tx->setFlags(tx->flags() & ~Qt::ItemIsEditable);
  }
}

void MainWindow::transformsDoubleClicked(const QModelIndex &index) {
  QString transformName = index.data().toString();
  // Create a new transform node.
  auto *node = registry.getBuilder(index.data().toString())();
  scene->addItem(node);
}

void MainWindow::openFolderClicked() {
  QString folder = QFileDialog::getExistingDirectory(this, "Open Folder");
  if (folder.isEmpty())
    return;

  // Create entries for the files in the directory in the file explorer.
  fileModel->clear();
  QDirIterator it(folder, QDirIterator::Subdirectories);
  while (it.hasNext()) {
    it.next();
    if (it.fileInfo().isFile()) {
      auto *item = new QStandardItem(it.fileInfo().absoluteFilePath());
      item->setFlags(item->flags() & ~Qt::ItemIsEditable);
      fileModel->appendRow(item);
    }
  }
}
