#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include "blockdatasource.h"
#include "blockmodel.h"
#include "portdatasource.h"
#include "portmodel.h"

int main(int argc, char *argv[])
{

    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QCoreApplication::setOrganizationName("TangibileDevices");
    QCoreApplication::setOrganizationDomain("tangibledevices.com");

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    qmlRegisterType<BlockDataSource>("com.company.models",1,0,"BlockDataSource");
    qmlRegisterType<BlockModel>("com.company.models",1,0,"BlockModel");
    qmlRegisterType<PortModel>("com.company.models",1,0,"PortModel");
    qmlRegisterType<PortDataSource>("com.company.models",1,0,"PortDataSource");


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
