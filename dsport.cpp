#include "dsport.h"

DSPort::DSPort(QObject *parent) : QObject(parent)
{
    qDebug()<<"Created: "<<this;

}

DSPort::~DSPort()
{
    qDebug()<<"Deleted: "<<this;
}

Port *DSPort::dsPort() const
{
    return m_dsPort;
}

void DSPort::setdsPort(Port *parentPort)
{
    if (m_dsPort == parentPort)
        return;

    m_dsPort = parentPort;
    emit dsPortChanged(m_dsPort);
}

void DSPort::startLink()
{
    beginInsertLink(m_dsPort->linkCount());
    m_dsPort->startLink();
    endInsertLink();
}

void DSPort::deleteLink(int linkIndex)
{
    beginRemoveLink(linkIndex);
    m_dsPort->removeLink(linkIndex);
    endRemoveLink();
}
