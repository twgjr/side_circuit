#ifndef EQUATION_H
#define EQUATION_H

#include <QObject>
#include "z3++.h"
#include <QDebug>
#include "equationparser.h"

class Equation : public QObject
{
    Q_OBJECT
public:
    explicit Equation(z3::context * context, QObject *parent = nullptr);

    QString getEquationString() ;
    void setEquationString(QString value);

    void printExprInfo();

    z3::expr getEquationExpression();
    void setEquationExpression(z3::expr equationExpression);

    void eqStrToExpr();

signals:

private:
    z3::context * m_equationContext;
    QString m_equationString = "";
    z3::expr m_equationExpression;
};

#endif // EQUATION_H
