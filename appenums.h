#ifndef APPENUMS_H
#define APPENUMS_H
#include <QObject>
#include <QtGlobal>

class DItemTypes : public QObject
{
    Q_OBJECT
public:
    enum Enum : int {
        BlockItem,
        Resistor
    };
    Q_ENUM(Enum)
};

class TestModes : public QObject
{
    Q_OBJECT
public:
    enum Enum : int {
        Test0,
        Test1
    };
    Q_ENUM(Enum)
};

#endif // APPENUMS_H
