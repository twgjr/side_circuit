import 'expression.dart';

//creates Expression AST's from string's of equations or functions
class Parser {
  Expression expressionGraph = Expression();  //points to the root of the abstract syntax tree

  // Order of ops for parsing equations with RegEx's
  Order order = Order();

  Parser() {
    Order order = Order();
  }

  void parseEquation(String equationString)
  {
    // clean up and simplify the string format
    equationString.replaceAll(" ", "" ); // remove spaces

    //go down the checklist
    assembleSyntaxTree(equationString,0,expressionGraph);
  }

  void assembleSyntaxTree(String equationString, int depth, Expression parentNode) {
    for( int i = 0 ; i < order.list.length; i++ ) {
      Map<String,String> element = order.list[i];
      String regExName = element.keys.first;
      RegExp regex = RegExp(element.values.first);

      if(regex.hasMatch(equationString)){
        String matchString = regex.stringMatch(equationString);
        RegExpMatch regExMatch = regex.firstMatch(equationString);
        int start = regExMatch.start;
        int end = regExMatch.end;

        bool breakForLoop = false;
        switch (regExName) {
          case "Parenthesis":{
            makeNodeBranchIn(equationString,matchString,depth,regExName,parentNode);
            breakForLoop = true;
            break;
          }
          case "Equals":
          case "LTOE":
          case "GTOE":
          case "LessThan":
          case "GreaterThan":
          case "NotEquals":
          case "Power":
          case "Multiply":
          case "Divide":
          case "Add":
          case "Subtract":{
            makeNodeBranchOut(equationString,matchString,start,end,depth,regExName,parentNode);
            breakForLoop = true;
            break;
          }
          case "Variable":
          case "Constant":{
            makeLeaf(matchString,regExName,parentNode);
            breakForLoop = true;
            break;
          }
        }
        if(breakForLoop){
          break;
        }
      }
    };
  }

  void makeLeaf(String matchString, String type, Expression parentNode)
  {
    Expression thisExpr = Expression();
    thisExpr.type = type;
    thisExpr.parent = parentNode;
    parentNode.children.add(thisExpr);

    if(type == "Variable"){
      thisExpr.varName = matchString;
    }
    if (type == "Constant") {
      thisExpr.value = num.parse(matchString);
    }
    return;
  }

  void makeNodeBranchOut(String equationString, String matchString, int start,
      int end, int depth, String type, Expression parentNode)
  {
    Expression thisExpr = Expression();
    thisExpr.type = type;
    thisExpr.parent = parentNode;
    parentNode.children.add(thisExpr);
    String sectionStr0=equationString.substring(0,start);
    assembleSyntaxTree(sectionStr0,depth+1,thisExpr);
    String sectionStr1=equationString.substring(end);
    assembleSyntaxTree(sectionStr1,depth+1,thisExpr);
    return;
  }

  void makeNodeBranchIn( String equationString, String matchString, int depth,
      String type, Expression parentNode)
  {
    Expression thisExpr = Expression();
    thisExpr.type = type;
    thisExpr.parent = parentNode;
    parentNode.children.add(thisExpr);
    String sectionStr=equationString.substring(2,-3);
    assembleSyntaxTree(sectionStr,depth+1,thisExpr);
    return;
  }

  void printTree(Expression expr) {
    //breadth first print children
    String spacer = "";
    for ( int ctr = 0 ; ctr < expr.depth() ; ctr++) {
      spacer += "->";
    }

    print("$spacer${expr.depth()},${expr.breadth()},${expr.type},${expr.varName},${expr.value}");

    for (Expression child in expr.children) {
      printTree(child);
    }
  }
}