#include "link.h"

Link::Link(QObject *parent) : QObject(parent),
    m_start(nullptr),
    m_end(nullptr),
    m_portConnected(false)
{
    //qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

Link::~Link()
{
    //qDebug()<<"Deleted: "<<this;
    disconnectEndPort();
}

void Link::removePoint(int index)
{
    if(!m_points.isEmpty()){
        m_points.remove(index);
    }
}

void Link::removeLastPoint()
{
    removePoint(m_points.size()-1);
}

QPointF Link::lastPoint() const
{
    if(!m_points.isEmpty()){
        return m_points[m_points.size()-1];
    }
    return QPointF();
}

void Link::setLastPoint(QPointF point)
{
    if(!m_points.isEmpty()){
        m_points[m_points.size()-1]=point;
    }
    emit lastPointChanged(point);
}

void Link::appendPoint(QPointF newPoint)
{
    if (lastPoint() == newPoint)
        return;

    m_points.append(newPoint);
    emit lastPointChanged(newPoint);
}

void Link::setEndPort(Port *endPort)
{
    if(m_end == endPort){
        return;
    }
    m_end = endPort;
    m_end->appendConnectedLink(this);

    connect(m_end,&Port::absPointChanged,this,[=](QPointF point){
        setLastPoint(point);
    });
}

void Link::disconnectEndPort()
{
    if(m_end){
        m_end->disconnect(this); //disconnect signals between end port and this link
        m_end->removeConnectedLink(this); //remove the Link* from the list in connected Port
        m_end=nullptr;
    }
}

Link *Link::thisLink()
{
    return this;
}

bool Link::portConnected() const
{
    return m_end?true:false;
}

Port *Link::startPort()
{
    return m_start;
}

void Link::setStartPort(Port *startPort)
{
    m_start = startPort;
}
