#ifndef ELEMENT_H
#define ELEMENT_H

#include <QObject>
#include <QHash>
#include "equation.h"
#include "block.h"

class Element : public QObject
{
    Q_OBJECT
public:

    Q_PROPERTY(int xPos READ xPos WRITE setXPos)
    Q_PROPERTY(int yPos READ yPos WRITE setYPos)
    Q_PROPERTY(Element* thisItem READ thisItem)
    Q_PROPERTY(int type READ type WRITE setType)
    Q_PROPERTY(int rotation READ rotation WRITE setRotation)

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

    explicit Element(int type, z3::context * context, QObject *parent = nullptr);

    // getters with qProperties
    int xPos() const;
    int yPos() const;

    //setters with qProperties
    void setXPos(int xPos);
    void setYPos(int yPos);

    Block *parentBlock() const;
    void setParentBlock(Block *parentBlock);

    // Ports
    Port * portAt( int portIndex );
    void addPort(int side, int position);
    void removePort(int portIndex);
    int portCount();

    Element* thisItem();

    int type() const;
    void setType(int type);

    int rotation() const;
    void setRotation(int rotation);

signals:
    // qAbstractItemModel signals
    void beginResetElement();
    void endResetElement();
    void beginInsertPort(int portIndex);
    void endInsertPort();
    void beginRemovePort(int portIndex);
    void endRemovePort();

private:
    int m_type;
    z3::context * m_elementContext;
    QHash<QString,double> m_properties;
    QVector<Equation*> m_equations;
    Block* m_parentBlock;
    int m_xPos;
    int m_yPos;
    int m_id;
    QVector<Port*> m_ports;
    int m_rotation;
};

#endif // ELEMENT_H
