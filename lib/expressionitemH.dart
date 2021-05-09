//#include "z3++.h"

class ExpressionItem
{
    String exprString="";
    int exprId=0;
    ExpressionItem parent;
    var children = [];//QVector<ExpressionItem*>
}