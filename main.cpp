#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include "datasource.h"
#include "diagrammodel.h"
#include "equationmodel.h"
#include "portmodel.h"
#include "linkmodel.h"
#include "resultmodel.h"
#include "elementmodel.h"

int main(int argc, char *argv[])
{

    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QCoreApplication::setOrganizationName("TangibileDevices");
    QCoreApplication::setOrganizationDomain("tangibledevices.com");

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    qmlRegisterType<DataSource>("com.company.models",1,0,"DataSource");
    qmlRegisterType<DiagramModel>("com.company.models",1,0,"DiagramModel");
    qmlRegisterType<EquationModel>("com.company.models",1,0,"EquationModel");
    qmlRegisterType<PortModel>("com.company.models",1,0,"PortModel");
    qmlRegisterType<LinkModel>("com.company.models",1,0,"LinkModel");
    qmlRegisterType<ResultModel>("com.company.models",1,0,"ResultModel");
    qmlRegisterType<ElementModel>("com.company.models",1,0,"ElementModel");

    const QUrl url(QStringLiteral("qrc:/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);
    engine.load(url);

    int ret;
    try {
        ret = app.exec();
    }  catch (...) {
        qDebug()<<"Uknown error occurred with application";
        return EXIT_FAILURE;
    }
    return ret;
}
