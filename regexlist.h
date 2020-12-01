#ifndef REGEXLIST_H
#define REGEXLIST_H

#include <QObject>
#include <QDebug>
#include <QRegularExpression>
#include <QRegularExpressionMatchIterator>
#include "expressionitem.h"

class RegExList : public QObject
{
    Q_OBJECT
public:
    explicit RegExList(QObject *parent = nullptr);

    struct RegExpr{
        QString exprCase;
        QString regExStr;
    };

    void initRegExList();
    RegExpr setRegExpr(QString exprCase,QString regExStr);
    QVector<RegExpr> formats() const;

private:
    QVector<RegExpr> m_formats;
};

#endif // REGEXLIST_H
