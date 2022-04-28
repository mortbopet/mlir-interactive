#include "mlir-viewer/mainwindow.h"
#include "ui_mainwindow.h"

#include <QAction>
#include <QDirIterator>
#include <QFileDialog>
#include <QFileSystemModel>
#include <QGraphicsView>
#include <QLabel>
#include <QListView>
#include <QSpacerItem>
#include <QStandardItemModel>
#include <QToolBar>
#include <QVBoxLayout>

#include <fstream>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include "mlir-viewer/PassExecuter.h"
#include "mlir-viewer/Scene.h"
#include "mlir-viewer/graph/MLIRModuleLoader.h"
#include "mlir-viewer/graph/TransformNode.h"

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

MainWindow::MainWindow(mlir::MLIRContext &context, TransformsRegistry &registry,
                       QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), registry(registry) {
  ui->setupUi(this);

  Q_INIT_RESOURCE(icons);

  executer = std::make_unique<PassExecuter>(context);
  scene = new Scene(*executer, this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  ui->graphicsView->setScene(scene);
  scene->setBackgroundBrush(QColor(Qt::gray));

  // Mock file explorer
  fileModel = new QFileSystemModel(this);
  ui->explorer->setModel(fileModel);
  fileModel->setRootPath(QDir::currentPath());
  ui->explorer->setRootIndex(fileModel->index(QDir::currentPath()));

  connect(ui->explorer, &QTreeView::doubleClicked, this,
          &MainWindow::explorerDoubleClicked);

  setupTransforms();
  setupActions();

  // Connect item focus changes to updates in the focus UI layout.
  connect(scene, &Scene::focusItem, this, [&](BaseItem *item) {
    clearLayout(ui->focusLayout);

    auto subLayout = new QVBoxLayout();
    ui->focusLayout->addLayout(subLayout);

    QString description = item->description();
    if (!description.isEmpty()) {
      auto *descriptionLabel = new QLabel(description);
      descriptionLabel->setWordWrap(true);
      subLayout->addWidget(descriptionLabel);
    }

    if (item)
      item->createUI(subLayout);

    // subLayout->addSpacing(1);
    ui->focusLayout->addSpacerItem(new QSpacerItem(
        1, 1, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding));
  });
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::explorerDoubleClicked(const QModelIndex &index) {
  QString filename = index.data(Qt::UserRole).toString();
  // Create a new source file node.
  QString path = fileModel->filePath(index);
  if (!QFileInfo(path).isFile())
    return;

  auto *node = new SourceFileNode(path, nullptr);
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

  saveAction = new QAction("Save", this);
  ui->menuFile->addAction(saveAction);
  connect(saveAction, &QAction::triggered, this,
          [this]() { this->saveClicked(); });
  saveAction->setShortcut(QKeySequence::Save);

  saveAsAction = new QAction("Save As", this);
  ui->menuFile->addAction(saveAsAction);
  connect(saveAsAction, &QAction::triggered, this, &MainWindow::saveAsClicked);
  saveAsAction->setShortcut(QKeySequence::SaveAs);
}

void MainWindow::setupTransforms() {
  auto *transformsModel = new QStandardItemModel(this);
  ui->transforms->setModel(transformsModel);
  connect(ui->transforms, &QListView::doubleClicked, this,
          &MainWindow::transformsDoubleClicked);

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

  ui->explorer->setRootIndex(fileModel->index(folder));
}

void MainWindow::saveClicked(QString filename) {
  if (filename.isEmpty()) {
    if (saveFile.isEmpty()) {
      return saveAsClicked();
    } else {
      filename = saveFile;
    }
  }

  std::ofstream os(filename.toStdString());
  cereal::JSONOutputArchive archive(os);
  archive(cereal::make_nvp("scene", *scene));
  os.close();
  saveFile = filename;
}

void MainWindow::saveAsClicked() {
  QString filename = QFileDialog::getSaveFileName(this, "Save File");
  if (filename.isEmpty())
    return;
  saveClicked(filename);
}
