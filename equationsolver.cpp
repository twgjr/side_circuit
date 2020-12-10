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

void EquationSolver::loadEquations(BlockItem *parentItem)
{
    //iterate through all children and load equations then recurse to children
    for (int i = 0 ; i < parentItem->childCount() ; i++) {
        if( (parentItem->child(i)->childCount() == 0) && (parentItem->parentItem() != nullptr) ){
            z3::expr expression = parentItem->equation()->getEquationExpression();
            registerEquation(expression);
            loadEquations(parentItem->child(i));
        }
    }
    //iterate through all children and load equations then recurse to children
    if(parentItem->parent() == nullptr){
        qDebug() << "ROOT";
    }
    for (int i = 0 ; i < parentItem->childCount() ; i++) {
        if( parentItem->child(i)->childCount() == 0 ){
            //is a leaf, then print and return, else continue to traverse the tree
            registerEquation(parentItem->child(i)->equation()->getEquationExpression());
        } else{
            loadEquations(parentItem->child(i));
        }
    }
}

void EquationSolver::solveEquations(BlockItem *parentItem)
{
    loadEquations(parentItem);

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
