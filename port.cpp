#include "port.h"

Port::Port(QObject *parent) : QObject(parent),
    m_itemParent(nullptr),
    m_side(0),
    m_position(0),
    m_name("label")
{
    qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

Port::~Port()
{
    //qDebug()<<"Deleted: "<<this;
    removeAllLinks();  //cleanup
}

void Port::setItemParent(DiagramItem *itemParent)
{
    m_itemParent = itemParent;
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
    newLink->setStartPort(this);
    emit beginInsertLink(m_links.count());
    m_links.append(newLink);
    emit endInsertLink();
}

void Port::removeLink(int linkIndex)
{
    emit beginRemoveLink(linkIndex);
    delete m_links[linkIndex];
    m_links.remove(linkIndex);
    emit endRemoveLink();
}

void Port::resetLinkModel()
{
    emit beginResetPort();
    emit endResetPort();
}

Port *Port::thisPort()
{
    return this;
}

QPointF Port::absPoint() const
{
    return m_absPoint;
}

void Port::setAbsPoint(QPointF absPoint)
{
    if (m_absPoint == absPoint)
        return;

    m_absPoint = absPoint;
    emit absPointChanged(m_absPoint);
}

QVector<Link*> Port::connectedLinks()
{
    return m_connectedLinks;
}

void Port::appendConnectedLink(Link *cLink)
{
    if(!m_connectedLinks.contains(cLink)){
        m_connectedLinks.append(cLink);
    }
}

void Port::removeConnectedLink(Link *cLink)
{
    if(m_connectedLinks.contains(cLink)){
        m_connectedLinks.remove(m_connectedLinks.indexOf(cLink));
    }
}

void Port::removeAllLinks()
{
    m_connectedLinks.clear();

}
