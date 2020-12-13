#include "link.h"

Link::Link(QObject *parent) : QObject(parent),
    m_start(nullptr),
    m_end(nullptr)
{
    qDebug()<<"Link created";
}
