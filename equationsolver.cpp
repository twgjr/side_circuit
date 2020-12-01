#include "equationsolver.h"

EquationSolver::EquationSolver(z3::context * context, QObject *parent) : QObject(parent),
    m_mainContext(context),
    m_optimizer(*context),
    m_solverModel(*context),
    m_z3solver(*context)
{
}

void EquationSolver::registerEquation(z3::expr z3Expr)
{
    m_z3solver.add(z3Expr);
}

void EquationSolver::solveEquations()
{
    switch (m_z3solver.check()) {
    case 0:{//UNSAT
        qDebug()<<"UNSAT!";
        break;
    }
    case 1:{//SAT
        printModel();
        break;
    }
    case 2:{//UKNOWN
        qDebug()<<"UNKNOWN!";
        break;
    }
    default:{
        qDebug()<<"No valid solver state!";
        break;
    }
    }
}

void EquationSolver::printModel()
{
    m_solverModel = m_z3solver.get_model();
    qDebug()<<"Got model. Traversing model.";
    qDebug() << "Model has " << QString::fromStdString(std::to_string(m_solverModel.size())) << " constants:";
    // traversing the model
    for (unsigned i = 0; i < m_solverModel.size(); i++) {
        QString name = QString::fromStdString(m_solverModel[i].name().str());
        qDebug()<<"Variable"<<name<<"at"<<i;
        qDebug()<<"    is const?: "<<m_solverModel[i].is_const();
        if(m_solverModel[i].is_const()){
            z3::func_decl fdecl = m_solverModel.get_const_decl(i);
            qDebug()<<"    get const declaration: "<<QString::fromStdString(fdecl.to_string());
            qDebug()<<"    get const value (interp): "<<QString::fromStdString(m_solverModel.get_const_interp(fdecl).to_string());
        }
        //qDebug()<<"    sort: "<<QString::fromStdString(m_solverModel[i].range().to_string()); // sort of the variable
    }
}
