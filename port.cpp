#include "port.h"

Port::Port(QObject *parent) : QObject(parent),
    m_blockParent(nullptr),
    m_link(nullptr),
    m_side(0),
    m_position(0),
    m_name(""),
    isConnected(false)
{
    //qDebug()<<"Port created";
}

Port::~Port()
{
    //qDebug()<<"Port destroyed.";
}

void Port::setBlockParent(BlockItem *blockParent)
{
    m_blockParent = blockParent;
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
