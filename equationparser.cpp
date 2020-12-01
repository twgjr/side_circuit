#include "equationparser.h"

EquationParser::EquationParser(z3::context * context, QObject *parent) : QObject(parent),
    m_context(context),
    m_z3Expr(*context)
{
    m_expressionGraph = new ExpressionItem(this);
}

void EquationParser::parseEquation(QString equationString)
{
    // clean up and simplify the string format
    equationString.replace( " ", "" ); // remove spaces

    //go down the checklist
    //qDebug()<<"Building AST";
    if(!assembleSyntaxTree(equationString,0,m_expressionGraph)){
        qDebug()<<"failed to parse!";
    }else{
        //qDebug()<<"Building z3 expression from AST";
        m_z3Expr = traverseSyntaxTree(m_expressionGraph->m_children[0]); //root is empty, skip to child
    }
}

bool EquationParser::assembleSyntaxTree(QString equationString,
                                        int depth,
                                        ExpressionItem * parentNode)
{
    for(int index = 1; index < m_regExList.formats().size() ; index++){
        QRegularExpression regex(m_regExList.formats()[index].regExStr);
        QRegularExpressionMatch match = regex.match(equationString);
        if(match.hasMatch()){
            QString matchString = match.captured(0);
            int start = match.capturedStart();
            int end = match.capturedEnd();
            int length = matchString.length();

            //no-match case 0 is skipped, checked in parseEquation() after complete parsing
            switch (index) {

            case 1:{// Parenthesis
                if((length)!=equationString.length()){
                    break;
                } else {
                    makeNodeBranchIn(equationString,matchString,depth,index,parentNode);
                    return true;
                }
            }
            case 2:{// Equals
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 3:{// Less than or Equals
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 4:{// Greater than or Equals
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 5:{// Less than
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 6:{ // Greater than
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 7:{ // Not Equal To
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 8:{ // Power
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 9:{// Multiply
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 10:{// Divide
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 11:{// Add
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 12:{ // Subtract
                makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                return true;
            }
            case 13:{// Variable
                makeLeaf(matchString,index,parentNode);
                return true;
            }
            case 14:{// Constant
                makeLeaf(matchString,index,parentNode);
                return true;
            }
            }
        }
    }
    if(equationString.length()==0){
        return false;
    }
    return false;
}

bool EquationParser::makeLeaf(QString matchString,
                              int id,
                              ExpressionItem * parentNode)
{
    //qDebug()<<"Level: "<<depth<<", adding leaf: "<<matchString;
    ExpressionItem * thisNode = new ExpressionItem(this);
    thisNode->m_string = matchString;
    thisNode->m_exprId = id;
    thisNode->m_parent = parentNode;
    parentNode->m_children.append(thisNode);
    return true;
}

bool EquationParser::makeNodeBranchOut(QString equationString,
                                       QString matchString,
                                       int start,
                                       int end,
                                       int depth,
                                       int id,
                                       ExpressionItem * parentNode)
{
    //qDebug()<<"Level: "<<depth<<", adding node: "<<matchString<<", branching outside";
    ExpressionItem * thisNode = new ExpressionItem(this);
    thisNode->m_string = matchString;
    thisNode->m_exprId = id;
    thisNode->m_parent = parentNode;
    parentNode->m_children.append(thisNode);
    QString sectionStr0=equationString.section("",0,start);
    assembleSyntaxTree(sectionStr0,depth+1,thisNode);
    QString sectionStr1=equationString.section("",end+1);
    assembleSyntaxTree(sectionStr1,depth+1,thisNode);
    return true;
}

bool EquationParser::makeNodeBranchIn(QString equationString,
                                      QString matchString,
                                      int depth,
                                      int id,
                                      ExpressionItem *parentNode)
{
    //qDebug()<<"Level: "<<depth<<", adding node: "<<matchString<<", branching inside";
    ExpressionItem * thisNode = new ExpressionItem(this);
    thisNode->m_string = matchString;
    thisNode->m_exprId = id;
    thisNode->m_parent = parentNode;
    parentNode->m_children.append(thisNode);
    QString sectionStr=equationString.section("",2,-3);
    qDebug()<<"sectioned string: "<<sectionStr;
    assembleSyntaxTree(sectionStr,depth+1,thisNode);
    return true;
}

z3::expr EquationParser::traverseSyntaxTree(ExpressionItem *parentNode)
{
    z3::expr exprBuffer(*m_context);
    //no-match case 0 is skipped, checked in parseEquation() after complete parsing
    switch (parentNode->m_exprId) {

    case 1:{// Parenthesis
        exprBuffer = ( traverseSyntaxTree(parentNode->m_children[0]) );
        break;
    }
    case 2:{// Equals
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        ==
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 3:{// Less than or Equals
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        <=
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 4:{// Greater than or Equals
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        >=
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 5:{// Less than
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        <
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 6:{ // Greater than
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        >
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 7:{ // Not Equal To
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        !=
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 8:{ // Power
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        ^
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 9:{// Multiply
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        *
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 10:{// Divide
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        /
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 11:{// Add
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        +
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 12:{ // Subtract
        exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                        -
                        traverseSyntaxTree(parentNode->m_children[1]);
        break;
    }
    case 13:{// Variable (a z3 constant)
        exprBuffer = m_context->real_const(parentNode->m_string.toUtf8());
        break;
    }
    case 14:{// Constant (a z3 value)
        std::string str = parentNode->m_string.toStdString();
        const char * value = str.c_str();
        exprBuffer = m_context->real_val(value);
        break;
    }
    }
    return exprBuffer;
}

ExpressionItem *EquationParser::expressionGraph()
{
    return m_expressionGraph;
}

z3::expr EquationParser::z3Expr()
{
    return m_z3Expr;
}

QString EquationParser::concatGraph(ExpressionItem *expressionItem){
    QString string="";
    if(expressionItem->m_children.size() == 0){
        return expressionItem->m_string;
    }
    if(expressionItem->m_children.size() == 1){
        string += expressionItem->m_string.section("",1,1);
        string += concatGraph(expressionItem->m_children[0]);
        string += expressionItem->m_string.section("",-2,-2);
    }
    if(expressionItem->m_children.size() == 2){
        string += concatGraph(expressionItem->m_children[0]);
        string += expressionItem->m_string;
        string += concatGraph(expressionItem->m_children[1]);
    }
    return string;
}
