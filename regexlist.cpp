#include "regexlist.h"

RegExList::RegExList(QObject *parent) : QObject(parent)
{
    initRegExList();
    //qDebug()<<"RegExList created with "<<m_formats.size()<<"items";
}

void RegExList::initRegExList()
{
    // 0
    m_formats.append(setRegExpr("NO_MATCH",""));
    // 1
    m_formats.append(setRegExpr("Parenthesis","\\(([^)]+)\\)"));
    // 2
    m_formats.append(setRegExpr("Equals","\\=="));
    // 3
    m_formats.append(setRegExpr("Less Than Or Equals","\\<="));
    // 4
    m_formats.append(setRegExpr("Greater Than Or Equals","\\>="));
    // 5
    m_formats.append(setRegExpr("Less Than","\\<"));
    // 6
    m_formats.append(setRegExpr("Greater Than","\\>"));
    // 7
    m_formats.append(setRegExpr("Not Equal To","\\!="));
    // 8
    m_formats.append(setRegExpr("Power","\\^"));
    // 9
    m_formats.append(setRegExpr("Multiply","\\*"));
    // 10
    m_formats.append(setRegExpr("Divide","\\/"));
    // 11
    m_formats.append(setRegExpr("Add","\\+"));
    // 12
    m_formats.append(setRegExpr("Subtract","\\-"));
    // 13
    m_formats.append(setRegExpr("Variable","((?=[^\\d])\\w+)")); // variable alphanumeric, not numberic alone
    // 14
    //m_formats.append(setRegExpr("Constant","\\d+"));   // just numeric (decimal and
    m_formats.append(setRegExpr("Constant","^[+-]?((\\d+(\\.\\d+)?)|(\\.\\d+))$"));
}

RegExList::RegExpr RegExList::setRegExpr(QString exprCase, QString regExStr)
{
    RegExpr regExpr;
    regExpr.exprCase = exprCase;
    regExpr.regExStr = regExStr;
    return regExpr;
}

QVector<RegExList::RegExpr> RegExList::formats() const
{
    return m_formats;
}
