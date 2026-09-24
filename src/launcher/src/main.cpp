#include "launcherTheme.h"
#include "mainDialog.h"

#include <QApplication>
#include <QIcon>

int main(int argc, char* argv[]) {
	QApplication a(argc, argv);
	QApplication::setWindowIcon(QIcon(":/icons/kytyps5.png"));
	LauncherTheme::Initialize(a);

	MainDialog w;

	w.emit Start();

	w.show();

	return QApplication::exec();
}
