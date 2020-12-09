#ifndef EXPRESSIONITEM_H
#define EXPRESSIONITEM_H

#include <QObject>
#include "regexlist.h"
#include "z3++.h"

class ExpressionItem : public QObject
{
    Q_OBJECT
public:
    explicit ExpressionItem(QObject *parent = nullptr);
    ~ExpressionItem();

    QString m_string="";
    int m_exprId=0;
    ExpressionItem * m_parent;
    QVector<ExpressionItem*> m_children;

private:

};

#endif // EXPRESSIONITEM_H
