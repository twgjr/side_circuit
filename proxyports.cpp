#include "proxyports.h"

ProxyPorts::ProxyPorts(QObject *parent) : QObject(parent)
{

}

BlockItem *ProxyPorts::parentBlock()
{
    return m_parentBlock;
}

void ProxyPorts::setParentBlock(BlockItem *block)
{
    m_parentBlock = block;
}

void ProxyPorts::addPort(int side, int position)
{
    beginInsertPort(m_parentBlock->portCount());
    m_parentBlock->addPort(side,position);
    endInsertPort();
}

void ProxyPorts::deletePort(int portIndex)
{
    beginRemovePort(portIndex);
    m_parentBlock->removePort(portIndex);
    endRemovePort();
}
