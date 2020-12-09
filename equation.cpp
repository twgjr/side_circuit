#include "equation.h"

Equation::Equation(z3::context * context, QObject *parent) : QObject(parent),
    m_equationContext(context),
    m_equationString(""),
    m_equationExpression(*context)
{
}

QString Equation::getEquationString()
{
    return m_equationString;
}

void Equation::setEquationString( QString value)
{
    m_equationString = value;
}

void Equation::printExprInfo()
{
    qDebug()<<"Expression info...";
    qDebug()<<"                id: "<<m_equationExpression.id();
    qDebug()<<"              kind: "<<m_equationExpression.kind();
    qDebug()<<"              hash: "<<m_equationExpression.hash();
    qDebug()<<"    Number of args: "<<m_equationExpression.num_args();
    for(unsigned i = 0 ; i< m_equationExpression.num_args() ; i++){
        qDebug()<<"        Arg " << i <<" : "<<QString::fromStdString(m_equationExpression.arg(i).to_string());
    }
    qDebug()<<"Function declaration info...";
    qDebug()<<"                id: "<<m_equationExpression.decl().id();
    qDebug()<<"              name: "<<QString::fromStdString(m_equationExpression.decl().name().str());
    qDebug()<<"             arity: "<<m_equationExpression.decl().arity();
    qDebug()<<"             range: "<<QString::fromStdString(m_equationExpression.decl().range().to_string());
    qDebug()<<"            domain: "<<QString::fromStdString(m_equationExpression.decl().domain(0).to_string());
    qDebug()<<"          is const: "<<m_equationExpression.decl().is_const();
    qDebug()<<"              kind: "<<m_equationExpression.decl().decl_kind();
}

z3::expr Equation::getEquationExpression()
{
    return m_equationExpression;
}

void Equation::setEquationExpression(z3::expr equationExpression)
{
    m_equationExpression = equationExpression;
}

void Equation::eqStrToExpr()
{
    EquationParser equationParser(m_equationContext);
    try {
        equationParser.parseEquation(m_equationString);
    }  catch (...) {
        qDebug()<<"Expression parsing error";
    }
    m_equationExpression = equationParser.z3Expr();
}
