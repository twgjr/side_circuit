#ifndef EQUATIONPARSER_H
#define EQUATIONPARSER_H

#include <QObject>
#include <QDebug>
#include "regexlist.h"
#include "expressionitem.h"
#include "z3++.h"

class EquationParser : public QObject
{
    Q_OBJECT
public:
    explicit EquationParser(z3::context * context, QObject *parent = nullptr);

    void parseEquation(QString equationString);//pre and post recursion processing

    bool assembleSyntaxTree(QString equationString,
                           int depth,
                           ExpressionItem * parentNode); //recursion for AST

    bool makeLeaf(QString matchString,
                  int id,
                  ExpressionItem * parentNode);

    bool makeNodeBranchOut(QString equationString,
                           QString matchString,
                           int start,
                           int end,
                           int depth,
                           int id,
                           ExpressionItem * parentNode);

    bool makeNodeBranchIn(QString equationString,
                          QString matchString,
                          int depth,
                          int id,
                          ExpressionItem * parentNode);

    z3::expr traverseSyntaxTree(ExpressionItem * parentNode);
    ExpressionItem * expressionGraph();
    z3::expr z3Expr();
    QString concatGraph(ExpressionItem * expressionItem);
    z3::expr getZeroExpression(ExpressionItem * parentNode);


private:
    ExpressionItem * m_expressionGraph;  //points to the root of the abstract syntax tree
    RegExList m_regExList;
    z3::context * m_context;
    z3::expr m_z3Expr;
};

#endif // EQUATIONPARSER_H
