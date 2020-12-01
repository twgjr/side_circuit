#ifndef EQUATIONSOLVER_H
#define EQUATIONSOLVER_H

#include <QObject>
#include <QDebug>
#include "z3++.h"
#include "equation.h"

class EquationSolver : public QObject
{
    Q_OBJECT
public:
    explicit EquationSolver(z3::context * context, QObject *parent = nullptr);

    void registerEquation(z3::expr z3Expr);
    void solveEquations();
    void printModel();
    void resetSolver();

//    QVector<Equation *> equationList() const;
//    void setEquationList(const QVector<Equation *> &equationList);
    //void appendEquation();
//    void clearEquationList();
    //TODO:
    // 1. function that sets constraint equations for variables extracted from equations
    // for example, upper (n>x) and lower bounds (x<n) and nominal/average/rms values
    // 2. function that generates multiple sets of equations to linearize control inputs for PSS analysis

signals:

private:
    z3::context * m_mainContext;
    QVector<Equation*> m_equationList;
    z3::optimize m_optimizer;
    z3::model m_solverModel;
    z3::solver m_z3solver;
};

#endif // EQUATIONSOLVER_H
