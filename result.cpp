#include "result.h"

Result::Result(z3::context * context, QObject *parent) : QObject(parent),
    m_equationContext(context),
    m_varName(""),
    m_varVal(0)
{

}

double Result::varVal() const
{
    qDebug()<<"varVal read is "<<m_varVal;
    return m_varVal;
}

void Result::setVarVal(double valNum)
{
    m_varVal = valNum;
    qDebug()<<"varVal write is "<<m_varVal;
}

QString Result::varName() const
{
    qDebug()<<"varName read is "<<m_varName;
    return m_varName;
}

void Result::setVarName(const QString &varString)
{
    m_varName = varString;
    qDebug()<<"varName write is "<<m_varName;
}
