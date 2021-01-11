#ifndef RESULT_H
#define RESULT_H

#include <QObject>
#include "equationsolver.h"

class Result : public QObject
{
    Q_OBJECT
public:
    //Q_PROPERTY(QString resultString READ resultString)
    Q_PROPERTY(double valNum READ valNum)
    Q_PROPERTY(QString varString READ varString)

    explicit Result(z3::context * context, QObject *parent = nullptr);

    QString varString() const;

    void setVarString(const QString &varString);

    double valNum() const;

    void setValNum(double valNum);

signals:
private:
    z3::context * m_equationContext;
    QString m_varString;
    double m_valNum;
};

#endif // RESULT_H
