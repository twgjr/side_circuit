class Order {
  List<Map<String,String>> list;
  Order(){
    list = [
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
      {"Variable":"((?=[^\\d])\\w+)"},// variable alphanumeric, not numeric alone
      {"Constant":"^[+-]?((\\d+(\\.\\d+)?)|(\\.\\d+))"}, // had to remove "?" from the end for dart
    ];
  }
}

class Solver{
  Order order = Order();


}

class Model{
  Order order = Order();
  List<Expression> truthList;  //list of solver objects that must evaluate to true
}

// the abstract syntax tree of constants, variables, and operators
class Expression {
  String string;  // @TODO factory for variables
  num value; // @TODO factory for constants
  String type;
  Expression parent;
  List<Expression> children;

  Expression(); //default expression constructor
}

class Parser
{
  Expression expressionGraph = Expression();  //points to the root of the abstract syntax tree

  // Order of ops for parsing equations with RegEx's
  Order order = Order();

  Parser();

  void parseEquation(String equationString)
  {
    // clean up and simplify the string format
    equationString.replaceAll(" ", "" ); // remove spaces

    //go down the checklist
    assembleSyntaxTree(equationString,0,expressionGraph);
  }

  void assembleSyntaxTree(String equationString, int depth, Expression parentNode)
  {
    order.list.forEach((element) {
      int orderId = order.list.indexOf(element);
      String regExName = element.keys.first;
      RegExp regex = RegExp(element.values.first);

      if(regex.hasMatch(equationString)){
        RegExpMatch regExMatch = regex.firstMatch(equationString);
        String matchString = regExMatch.toString();
        int start = regExMatch.start;
        int end = regExMatch.end;

        switch (regExName) {
          case "Parenthesis":{
            makeNodeBranchIn(equationString,matchString,depth,regExName,parentNode);
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
            break;
          }
          case "Variable":
          case "Constant":{
            makeLeaf(matchString,regExName,parentNode);
            break;
          }
        }
      }
    });
  }

  void makeLeaf(String matchString, String type, Expression parentNode)
  {
    Expression thisExpr = Expression();
    thisExpr.string = matchString;
    thisExpr.type = type;
    thisExpr.parent = parentNode;
    parentNode.children.add(thisExpr);
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
    String sectionStr1=equationString.substring(end+1);
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
}