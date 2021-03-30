#ifndef EQUATIONPARSER_H
#define EQUATIONPARSER_H

#include <QObject>
#include <QDebug>
#include "expressionitem.h"
#include "z3++.h"
#include <QRegularExpression>

class EquationParser : public QObject
{
    Q_OBJECT
public:

    //This enum controls the mathematical order of evaluation
    //for all parsing
    enum OrderOfOperations {
        Parenthesis,
        Equals,LTOE,GTOE,LessThan,GreaterThan,NotEquals,
        //Exponent,Logarithm,Sine,ASine,Cos,ACos,Tan,ATan,
        Power,Multiply,Divide,Add,Subtract,
        Variable,Constant
    };

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

    void initRegExList();

private:
    ExpressionItem * m_expressionGraph;  //points to the root of the abstract syntax tree
    z3::context * m_context;
    z3::expr m_z3Expr;
    QMap<int,QString> m_formats;
    QVector<QString> m_regExList;
};

#endif // EQUATIONPARSER_H
