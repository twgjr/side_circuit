#include "link.h"

Link::Link(QObject *parent) : QObject(parent),
    m_start(nullptr),
    m_end(nullptr),
    m_startX(0),
    m_startY(0),
    m_endX(0),
    m_endY(0)
{
    qDebug()<<"Created: "<<this;
}

Link::~Link()
{
    qDebug()<<"Deleted: "<<this;
}

int Link::startX() const
{
    return m_startX;
}

int Link::startY() const
{
    return m_startY;
}

int Link::endX() const
{
    return m_endX;
}

int Link::endY() const
{
    return m_endY;
}

void Link::setStartX(int startX)
{
    m_startX = startX;
}

void Link::setStartY(int startY)
{
    m_startY = startY;
}

void Link::setEndX(int endX)
{
    m_endX = endX;
}

void Link::setEndY(int endY)
{
    m_endY = endY;
}
