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
    qDebug()<<"added :"<<QString::fromStdString(z3Expr.to_string());
}

void EquationSolver::loadEquations(DiagramItem *rootItem)
{
    //load all equations for the current root in the recursion
    for ( int i = 0 ; i < rootItem->equationCount(); i++){
        z3::expr expression = rootItem->equationAt(i)->getEquationExpression();
        registerEquation(expression);
    }

    // visit all child blocks to load equations
    for (int i = 0 ; i < rootItem->childItemCount() ; i++) {
        if( rootItem->childItemAt(i)->childItemCount() == 0 ){
            //no more child blocks to visit
            return;
        } else {
            loadEquations(rootItem->childItemAt(i));
        }
    }
}

void EquationSolver::solveEquations(DiagramItem *parentItem)
{
    loadEquations(parentItem);

    z3::expr x = m_mainContext->real_const("x");
    m_z3solver.add(x!=0);

    switch (m_z3solver.check()) {
    case 0:{//UNSAT
        qDebug()<<"UNSAT!";
        break;
    }
    case 1:{//SAT
        printModel();
        updateResults(parentItem);
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
    }
}

void EquationSolver::updateResults(DiagramItem *rootItem)
{
    rootItem->clearResults();
    for (unsigned i = 0; i < m_solverModel.size(); i++) {
        QString name = QString::fromStdString(m_solverModel[i].name().str());
        if(m_solverModel[i].is_const()){
            z3::func_decl fdecl = m_solverModel.get_const_decl(i);
            QString resultString = QString::fromStdString(m_solverModel.get_const_interp(fdecl).get_decimal_string(3));
            bool doubleIsOK = false;
            double result = resultString.toDouble(&doubleIsOK);
            if(!doubleIsOK){
                qDebug()<<"Error converting result to double";
            }

            rootItem->addResult(name,result);
        }
    }
}
