#include "element.h"

Element::Element(int type, z3::context * context, QObject *parent) : QObject(parent),
    m_type(type),
    m_elementContext(context),
    m_parentBlock(nullptr),
    m_xPos(0),
    m_yPos(0),
    m_rotation(0)
{
    //qDebug()<<"created: "<<this;
    switch (type) {
    case Resistor:
        addPort(2,45);
        addPort(3,45);
        break;
    }
}

int Element::xPos() const
{
    return m_xPos;
}

int Element::yPos() const
{
    return m_yPos;
}

void Element::setXPos(int xPos)
{
    m_xPos = xPos;
}

void Element::setYPos(int yPos)
{
    m_yPos = yPos;
}

Block *Element::parentBlock() const
{
    return m_parentBlock;
}

void Element::setParentBlock(Block *parentBlock)
{
    m_parentBlock = parentBlock;
}

Port *Element::portAt(int portIndex)
{
    return m_ports[portIndex];
}

void Element::addPort(int side, int position)
{
    Port * newPort = new Port(this);
    Element * thisItem = static_cast<Element*>(this);
    newPort->setElementParent(thisItem);
    newPort->setSide(side);
    newPort->setPosition(position);
    emit beginInsertPort(m_ports.count());
    m_ports.append(newPort);
    emit endInsertPort();
}

void Element::removePort(int portIndex)
{
    qDebug()<<"port Index to be removed "<< portIndex;
    emit beginRemovePort(portIndex);
    delete m_ports[portIndex];
    m_ports.remove(portIndex);
    emit endRemovePort();
}

int Element::portCount()
{
    return m_ports.count();
}

Element *Element::thisItem()
{
    return this;
}

int Element::type() const
{
    return m_type;
}

void Element::setType(int type)
{
    m_type = type;
}

int Element::rotation() const
{
    return m_rotation;
}

void Element::setRotation(int rotation)
{
    m_rotation = rotation;
}
