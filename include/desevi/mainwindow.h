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

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private:
  void explorerDoubleClicked(const QModelIndex &index);
  void transformsDoubleClicked(const QModelIndex &index);
  void setupTransforms();
  void setupActions();
  void openFolderClicked();

  /// When enabled, the pipeline will be execute on every change.
  QAction *interactiveAction = nullptr;
  /// Executes the pipeline.
  QAction *runAction = nullptr;
  QAction *openFolderAction = nullptr;

  Ui::MainWindow *ui;
  Scene *scene;
  TransformsRegistry registry;
  QStandardItemModel *fileModel;
};
#endif // MAINWINDOW_H
