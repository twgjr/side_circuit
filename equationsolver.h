#ifndef EQUATIONSOLVER_H
#define EQUATIONSOLVER_H

#include <QObject>
#include <QDebug>
#include "z3++.h"
#include "equation.h"
#include "blockitem.h"

class EquationSolver : public QObject
{
    Q_OBJECT
public:
    explicit EquationSolver(z3::context * context, QObject *parent = nullptr);

    void registerEquation(z3::expr z3Expr);
    void loadEquations(BlockItem *parentItem);
    void solveEquations(BlockItem * parentItem);
    void printModel();
    void resetSolver();

signals:

private:
    z3::context * m_mainContext;
    QVector<Equation*> m_equationList;
    z3::optimize m_optimizer;
    z3::model m_solverModel;
    z3::solver m_z3solver;
};

#endif // EQUATIONSOLVER_H
