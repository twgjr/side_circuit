#ifndef ELEMENT_H
#define ELEMENT_H

#include <QObject>
#include <QHash>
#include "equation.h"

class Element : public QObject
{
    Q_OBJECT
public:
    enum ElementType{
        GenericElement,
        Resistor,
        Capacitor,
        Inductor,
        SolidStateSwitch,
        Diode,
        IC,
        LED,
        Battery,
        Connector
    };

    explicit Element(int type, QObject *parent = nullptr);

signals:
private:
    int m_type;
    QHash<QString,double> m_properies;
    QVector<Equation*> m_equations;
};

#endif // ELEMENT_H
