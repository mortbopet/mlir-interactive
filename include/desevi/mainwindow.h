#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "desevi/TransformsRegistry.h"
#include "desevi/graph/FileNode.h"

QT_BEGIN_NAMESPACE namespace Ui { class MainWindow; }
class QGraphicsScene;
class QAction;
class QStandardItemModel;
QT_END_NAMESPACE

class Scene;
class PassExecuter;

namespace mlir {
class MLIRContext;
}

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(mlir::MLIRContext &context, TransformsRegistry &registry,
             QWidget *parent = nullptr);
  ~MainWindow();

private:
  void explorerDoubleClicked(const QModelIndex &index);
  void transformsDoubleClicked(const QModelIndex &index);
  void setupTransforms();
  void setupActions();
  void openFolderClicked();
  void loadDirectory(const QString &path);

  /// When enabled, the pipeline will be execute on every change.
  QAction *interactiveAction = nullptr;
  /// Executes the pipeline.
  QAction *runAction = nullptr;
  QAction *openFolderAction = nullptr;

  Ui::MainWindow *ui;
  Scene *scene;
  QStandardItemModel *fileModel;
  TransformsRegistry &registry;
  std::unique_ptr<PassExecuter> executer;
};
#endif // MAINWINDOW_H
