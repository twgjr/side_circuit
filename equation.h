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
    Q_PROPERTY(int eqXPos READ eqXPos WRITE setEqXPos NOTIFY eqXPosChanged)
    Q_PROPERTY(int eqYPos READ eqYPos WRITE setEqYPos NOTIFY eqYPosChanged)


    explicit Equation(z3::context * context, QObject *parent = nullptr);

    QString getEquationString() ;
    void setEquationString(QString value);

    void printExprInfo();

    z3::expr getEquationExpression();
    void setEquationExpression(z3::expr equationExpression);

    void eqStrToExpr();

    int eqXPos() const;
    int eqYPos() const;

    void setEqXPos(int eqXPos);
    void setEqYPos(int eqYPos);

signals:
    void eqXPosChanged(int eqXPos);
    void eqYPosChanged(int eqYPos);

private:
    z3::context * m_equationContext;
    QString m_equationString = "";
    z3::expr m_equationExpression;
    int m_eqXPos;
    int m_eqYPos;
};

#endif // EQUATION_H
