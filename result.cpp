#include "result.h"

Result::Result(z3::context * context, QObject *parent) : QObject(parent),
    m_equationContext(context),
    m_varString(""),
    m_valNum(0)
  //m_resultString("")
{

}

double Result::valNum() const
{
    return m_valNum;
}

void Result::setValNum(double valNum)
{
    m_valNum = valNum;
}

QString Result::varString() const
{
    return m_varString;
}

void Result::setVarString(const QString &varString)
{
    m_varString = varString;
}
