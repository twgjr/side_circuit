#include "equation.h"

Equation::Equation(z3::context * context, QObject *parent) : QObject(parent),
    m_equationContext(context),
    m_type("equation"),
    m_equationString(""),
    m_equationExpression(*context),
    m_xPos(0),
    m_yPos(0),
    m_itemWidth(0),
    m_itemHeight(0)
{
    qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

Equation::~Equation()
{
    qDebug()<<"Destroyed: "<<this;
}

void Equation::setEquationString( QString value)
{
    m_equationString = value;
    eqStrToExpr();
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

int Equation::xPos() const
{
    return m_xPos;
}

int Equation::yPos() const
{
    return m_yPos;
}

void Equation::setXPos(int eqXPos)
{
    if (m_xPos == eqXPos)
        return;

    m_xPos = eqXPos;
    emit xPosChanged(m_xPos);
}

void Equation::setYPos(int eqYPos)
{
    if (m_yPos == eqYPos)
        return;

    m_yPos = eqYPos;
    emit yPosChanged(m_yPos);
}

QString Equation::type() const
{
    return m_type;
}

void Equation::setType(QString type)
{
    if (m_type == type)
        return;

    m_type = type;
    emit typeChanged(m_type);
}

QString Equation::equationString() const
{
    return m_equationString;
}

int Equation::itemWidth() const
{
    return m_itemWidth;
}

int Equation::itemHeight() const
{
    return m_itemHeight;
}

void Equation::setItemWidth(int itemWidth)
{
    if (m_itemWidth == itemWidth)
        return;

    m_itemWidth = itemWidth;
    emit itemWidthChanged(m_itemWidth);
}

void Equation::setItemHeight(int itemHeight)
{
    if (m_itemHeight == itemHeight)
        return;

    m_itemHeight = itemHeight;
    emit itemHeightChanged(m_itemHeight);
}
