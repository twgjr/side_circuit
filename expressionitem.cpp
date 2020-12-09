#include "expressionitem.h"

ExpressionItem::ExpressionItem(QObject *parent) : QObject(parent)
{
    //qDebug()<<"ExpressionItem created";
}

ExpressionItem::~ExpressionItem()
{
    //When the destructor is called, it must delete each
    // of these to ensure that their memory is reused
    qDeleteAll(m_children);
}

