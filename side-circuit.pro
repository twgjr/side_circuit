QT += quick

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        datasource.cpp \
        diagramitem.cpp \
        diagrammodel.cpp \
        equation.cpp \
        equationmodel.cpp \
        equationparser.cpp \
        equationsolver.cpp \
        expressionitem.cpp \
        link.cpp \
        linkmodel.cpp \
        main.cpp \
        port.cpp \
        portmodel.cpp \
        result.cpp \
        resultmodel.cpp

RESOURCES += qml.qrc \
    javascript.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Additional import path used to resolve QML modules just for Qt Quick Designer
QML_DESIGNER_IMPORT_PATH =

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    appenums.h \
    datasource.h \
    diagramitem.h \
    diagrammodel.h \
    equation.h \
    equationmodel.h \
    equationparser.h \
    equationsolver.h \
    expressionitem.h \
    link.h \
    linkmodel.h \
    port.h \
    portmodel.h \
    result.h \
    resultmodel.h

win32: LIBS += -L$$PWD/../z3/z3-4.8.10-x64-win/bin/ -llibz3

INCLUDEPATH += $$PWD/../z3/z3-4.8.10-x64-win/include
DEPENDPATH += $$PWD/../z3/z3-4.8.10-x64-win/include

win32:!win32-g++: PRE_TARGETDEPS += $$PWD/../z3/z3-4.8.10-x64-win/bin/libz3.lib
