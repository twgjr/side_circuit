QT += quick

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        blockdatasource.cpp \
        blockitem.cpp \
        blockmodel.cpp \
        equation.cpp \
        equationparser.cpp \
        equationsolver.cpp \
        expressionitem.cpp \
        link.cpp \
        main.cpp \
        port.cpp \
        portdatasource.cpp \
        portmodel.cpp \
        regexlist.cpp

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

win32: LIBS += -L$$PWD/../libbuilds/z3/z3-4.8.9-x64-win/bin/ -llibz3

INCLUDEPATH += $$PWD/../libbuilds/z3/z3-4.8.9-x64-win/include
DEPENDPATH += $$PWD/../libbuilds/z3/z3-4.8.9-x64-win/include

win32:!win32-g++: PRE_TARGETDEPS += $$PWD/../libbuilds/z3/z3-4.8.9-x64-win/bin/libz3.lib
else:win32-g++: PRE_TARGETDEPS += $$PWD/../libbuilds/z3/z3-4.8.9-x64-win/bin/liblibz3.a

HEADERS += \
    blockdatasource.h \
    blockitem.h \
    blockmodel.h \
    equation.h \
    equationparser.h \
    equationsolver.h \
    expressionitem.h \
    link.h \
    port.h \
    portdatasource.h \
    portmodel.h \
    regexlist.h

DISTFILES +=
