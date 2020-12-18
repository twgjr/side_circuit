#include "portdatasource.h"

PortDataSource::PortDataSource(QObject *parent) : QObject(parent)
{

}

BlockItem *PortDataSource::parentBlock()
{
    return m_parentBlock;
}

void PortDataSource::setParentBlock(BlockItem *block)
{
    m_parentBlock = block;
}

void PortDataSource::addPort(int side, int position)
{
    beginInsertPort(m_parentBlock->portCount());
    m_parentBlock->addPort(side,position);
    endInsertPort();
}

void PortDataSource::deletePort(int portIndex)
{
    beginRemovePort(portIndex);
    m_parentBlock->removePort(portIndex);
    endRemovePort();}
/*
int PortDataSource::portCount()
{
    return m->portCount();
}
*/

/*
int PortDataSource::portSide(int portNum)
{
    return m_proxyRoot->proxyChild(blockIndex)->portSide(portNum);
}
*/
/*
int PortDataSource::portPosition(int portNum)
{
    return m_proxyRoot->proxyChild(blockIndex)->portPosition(portNum);
}
*/
