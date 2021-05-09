import 'expressionitemH.dart';
import 'dart:ffi' as ffi;
#include "z3++.h"

enum OrderOfOps {
  Parenthesis,
  Equals,LTOE,GTOE,LessThan,GreaterThan,NotEquals,
  //Exponent,Logarithm,Sine,ASine,Cos,ACos,Tan,ATan,
  Power,Multiply,Divide,Add,Subtract,
  Variable,Constant
}

class EquationParser
{
    ExpressionItem expressionGraph = ExpressionItem();  //points to the root of the abstract syntax tree
    z3::context * m_context;
    z3::expr m_z3Expr;

    // Order of ops for parsing equations with RegEx's
    List<Map<String,String>> formats = [
        {"Parenthesis":"\\(([^)]+)\\)"},
        {"Equals":"\\=="},
        {"LTOE":"\\<="},
        {"GTOE":"\\>="},
        {"LessThan":"\\<"},
        {"GreaterThan":"\\>"},
        {"NotEquals":"\\!="},
        {"Power":"\\^"},
        {"Multiply":"\\*"},
        {"Divide":"\\/"},
        {"Add":"\\+"},
        {"Subtract":"\\-"},
        {"Variable":"((?=[^\\d])\\w+)"}, // variable alphanumeric, not numeric alone
        {"Constant":"^[+-]?((\\d+(\\.\\d+)?)|(\\.\\d+))"},
    ];

    EquationParser(/*z3::context * context*/);

    void parseEquation(String equationString)
    {
      // clean up and simplify the string format
      equationString.replaceAll(" ", "" ); // remove spaces

      //go down the checklist
      if(!assembleSyntaxTree(equationString,0,expressionGraph)){
        print("failed to parse!");
      }else{
        m_z3Expr = traverseSyntaxTree(expressionGraph.children[0]); //root is empty, skip to child
      }
    }

    bool assembleSyntaxTree(String equationString, int depth, ExpressionItem parentNode)
    {
        formats.forEach((element) {
            int index = formats.indexOf(element);
            String regExName = formats[index].keys.first;
            RegExp regex = RegExp(formats[index].values.first);

            if(regex.hasMatch(equationString)){
                String matchString = regex.stringMatch(equationString);
                int start = match.capturedStart();
                int end = match.capturedEnd();
                int length = matchString.length;

                //no-match case 0 is skipped, checked in parseEquation() after complete parsing
                switch (regExName) {

                    case "Parenthesis":{
                        if((length)!=equationString.length){
                            break;
                        } else {
                            makeNodeBranchIn(equationString,matchString,depth,index,parentNode);
                            return true;
                        }
                        break;
                    }
                    case "Equals":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "LTOE":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "GTOE":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "LessThan":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "GreaterThan":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "NotEquals":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "Power":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "Multiply":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "Divide":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "Add":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "Subtract":{
                        makeNodeBranchOut(equationString,matchString,start,end,depth,index,parentNode);
                        return true;
                    }
                    case "Variable":{
                        makeLeaf(matchString,index,parentNode);
                        return true;
                    }
                    case "Constant":{
                        makeLeaf(matchString,index,parentNode);
                        return true;
                    }
                }
            }
        });

        if(equationString.length==0){
            return false;
        }
        return false;
    }

    bool makeLeaf(String matchString, int id, ExpressionItem parentNode)
    {
        ExpressionItem thisNode = ExpressionItem();
        thisNode->m_string = matchString;
        thisNode->m_exprId = id;
        thisNode->m_parent = parentNode;
        parentNode->m_children.append(thisNode);
        return true;
    }

    bool makeNodeBranchOut(String equationString, String matchString, int start,
        int end, int depth, int id, ExpressionItem parentNode)
    {
        ExpressionItem thisNode = ExpressionItem(this);
        thisNode->m_string = matchString;
        thisNode->m_exprId = id;
        thisNode->m_parent = parentNode;
        parentNode->m_children.append(thisNode);
        String sectionStr0=equationString.section("",0,start);
        assembleSyntaxTree(sectionStr0,depth+1,thisNode);
        String sectionStr1=equationString.section("",end+1);
        assembleSyntaxTree(sectionStr1,depth+1,thisNode);
        return true;
    }

    bool makeNodeBranchIn( String equationString, String matchString, int depth,
        int id, ExpressionItem parentNode)
    {
        ExpressionItem thisNode = ExpressionItem();
        thisNode->m_string = matchString;
        thisNode->m_exprId = id;
        thisNode->m_parent = parentNode;
        parentNode->m_children.append(thisNode);
        String sectionStr=equationString.section("",2,-3);
        print("sectioned string: $(sectionStr)");
        assembleSyntaxTree(sectionStr,depth+1,thisNode);
        return true;
    }

    z3::expr traverseSyntaxTree(ExpressionItemparentNode)
    {
        z3::expr exprBuffer(*m_context);

        switch (parentNode->m_exprId) {

            case OrderOfOps.Parenthesis:{
                exprBuffer = ( traverseSyntaxTree(parentNode->m_children[0]) );
                break;
            }
            case OrderOfOps.Equals:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                ==
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.LTOE:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                <=
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.GTOE:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                >=
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.LessThan:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                <
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.GreaterThan:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                >
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.NotEquals:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                !=
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.Power:{
                exprBuffer =    z3::pw(traverseSyntaxTree(parentNode->m_children[0]),
                traverseSyntaxTree(parentNode->m_children[1]));
                break;
            }
            case OrderOfOps.Multiply:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                *
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.Divide:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                /
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.Add:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                +
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.Subtract:{
                exprBuffer =    traverseSyntaxTree(parentNode->m_children[0])
                -
                traverseSyntaxTree(parentNode->m_children[1]);
                break;
            }
            case OrderOfOps.Variable:{
                exprBuffer = m_context->real_const(parentNode->m_string.toUtf8());
                break;
            }
            case OrderOfOps.Constant:{
                std::string str = parentNode->m_string.toStdString();
                const char * value = str.c_str();
                exprBuffer = m_context->real_val(value);
                break;
            }
        }
        return exprBuffer;
    }

    z3::expr z3Expr()
    {
      return m_z3Expr;
    }

    String concatGraph(ExpressionItem expressionItem){
        String string="";
        if(expressionItem.children.length == 0){
            return expressionItem.exprString;
        }
        if(expressionItem.children.length == 1){
            string += expressionItem->m_string.section("",1,1);
            string += concatGraph(expressionItem->m_children[0]);
            string += expressionItem->m_string.section("",-2,-2);
        }
        if(expressionItem.children.length == 2){
            string += concatGraph(expressionItem->m_children[0]);
            string += expressionItem->m_string;
            string += concatGraph(expressionItem->m_children[1]);
        }
        return string;
    }
}
