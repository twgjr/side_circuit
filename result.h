#ifndef RESULT_H
#define RESULT_H

#include <QObject>
#include "equationsolver.h"
#include <QDebug>

class Result : public QObject
{
    Q_OBJECT
public:
    Q_PROPERTY(QString varName READ varName)
    Q_PROPERTY(double varVal READ varVal)

    explicit Result(z3::context * context, QObject *parent = nullptr);

    QString varName() const;

    void setVarName(const QString &varName);

    double varVal() const;

    void setVarVal(double varVal);

private:
    z3::context * m_equationContext;
    QString m_varName;
    double m_varVal;
};

#endif // RESULT_H
