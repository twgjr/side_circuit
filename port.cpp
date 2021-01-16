#include "port.h"

Port::Port(QObject *parent) : QObject(parent),
    m_blockParent(nullptr),
    m_elementParent(nullptr),
    m_side(0),
    m_position(0),
    m_name("label"),
    isConnected(false),
    m_state(PortModel::NotConnected)
{
    //qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

Port::~Port()
{
    //qDebug()<<"Deleted: "<<this;
}

void Port::setBlockParent(Block *blockParent)
{
    m_blockParent = blockParent;
}

void Port::setConnectedLink(Link *connectedLink)
{
    m_connectedLinks.append(connectedLink);
}

void Port::setSide(int side)
{
    m_side = side;
}

void Port::setPosition(int position)
{
    m_position = position;
}

int Port::side() const
{
    return m_side;
}

int Port::position() const
{
    return m_position;
}

QString Port::name() const
{
    return m_name;
}

void Port::setName(QString name)
{
    m_name = name;
}

QVector<Link *> Port::links()
{
    return m_links;
}

Link *Port::linkAt(int linkIndex)
{
    return m_links[linkIndex];
}

int Port::linkCount()
{
    return m_links.count();
}

void Port::startLink()
{
    Link * newLink = new Link(this);
    emit beginInsertLink(m_links.count());
    m_links.append(newLink);
    emit endInsertLink();
}

void Port::removeLink(int linkIndex)
{
    qDebug()<<"link Index to be removed "<<linkIndex;
    emit beginRemoveLink(linkIndex);
    delete m_links[linkIndex];
    m_links.remove(linkIndex);
    emit endRemoveLink();
}

Port *Port::thisPort()
{
    return this;
}

void Port::setThisPort(Port *thisPort)
{
    if (m_thisPort == thisPort)
        return;

    m_thisPort = thisPort;
    emit thisPortChanged(m_thisPort);
}

int Port::state() const
{
    return m_state;
}

void Port::setState(int state)
{
    if (m_state == state)
        return;

    m_state = state;
    emit stateChanged(m_state);
}

void Port::setElementParent(Element *elementParent)
{
    m_elementParent = elementParent;
}


