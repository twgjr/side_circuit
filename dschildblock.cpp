#include "dschildblock.h"

DSChildBlock::DSChildBlock(QObject *parent) : QObject(parent)
{
    qDebug()<<"Created: "<<this;
}

DSChildBlock::~DSChildBlock()
{
    qDebug()<<"Deleted: "<<this;
}

BlockItem *DSChildBlock::dsChildBlock()
{
    return m_dsChildBlock;
}

void DSChildBlock::setdsChildBlock(BlockItem *block)
{
    m_dsChildBlock = block;
}

Port *DSChildBlock::port(int portIndex)
{
    return m_dsChildBlock->ports()[portIndex];
}

void DSChildBlock::addPort(int side, int position)
{
    beginInsertPort(m_dsChildBlock->portCount());
    m_dsChildBlock->addPort(side,position);
    endInsertPort();
}

void DSChildBlock::deletePort(int portIndex)
{
    beginRemovePort(portIndex);
    m_dsChildBlock->removePort(portIndex);
    endRemovePort();//crashes here with "index out of range"
}

void DSChildBlock::startLink(int portIndex)
{
    m_dsChildBlock->ports()[portIndex]->startLink();
}
