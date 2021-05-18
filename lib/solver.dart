enum SolverState {
  Unknown,
  Sat,
  Unsat
}

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

// Model:  the top level list of expression roots for the base declarations of the
// mathematical model.  For example, dx/dt object would be declared, but does
// not store the entire set of incremental sets of equations required to solve
// a transient, time-valued system.  If the system is non-differential, the
// the solver will only run with the base set. If the system contains/enabled
// differentials, then it will generated enough equation sets to solve the
// transient from one steady state to another.
// During differential solve, there will be a new list of variables for each
class Model {
  Order order = Order();
  List<Expression> expressionList; // single list of Expression roots for the Model

  int solve() {
    bool decided = false;
    var solverState = SolverState.Unknown;
    while (!decided) {
      switch (solverState) {
        case SolverState.Unknown:
          {
            solverState = checkModel()?SolverState.Sat:SolverState.Unsat;
            break;
          }
        case SolverState.Sat:
          {
            print("Solution is sat.");
            decided = true;
            break;
          }
        case SolverState.Unsat:
          {
            print("Solution is UNSAT!");
            decided = true;
            break;
          }
      }
    }
    return solverState.index;
  }

  bool checkModel(){
    bool modelIsSat = true;
    /*
    @TODO
     set variable assignments with single level variable with constant and
     continue until no further simplification can be made
     set the range of possible values for remaining variables (-inf,+inf, or a finite range)
     Use bisection to split solution ranges for variables and branch
     heuristic to choose which expression to branch next is the simplest expression
     with variables appearing in the most expressions. This heuristic simplifies the
     complexity of the recursion the earliest and fastest.
     implement SAT/UNSAT based on DPLL algorithm.
    */


    expressionList.forEach((exprRoot) {
      if(!exprRoot.isSat){ //skip any that already found solution and are independent
        modelIsSat &= checkBoolean(exprRoot,true);
      }
    });
    //modelIsSat is false if any single root Expression is false;
    return modelIsSat;
  }

  // initial pass through should remove unit literals, then pure literals,
  // unit literal is a variable that has an exact assignment such as x=2 at the top level.
  // pure literals are variables such that the range of values is set x<2 and x>0.
  // The literals should be used to restrict the number of branches for the algorithm.
  bool checkBoolean(Expression expr, bool isInitialGuess) {
    switch (expr.type) {
      case "Parenthesis":
        checkBoolean(expr.children[0],false);
        break;
      case "Equals":
        {
          if (expr.children[0].value == expr.children[1].value) {
            return true;
          } else {
            //change child and recurse child
            if(expr.children[0].isSat){
              expr.children[1].value = expr.children[0].value;
              chooseChildValue(expr.children[1]);
            } else if(expr.children[1].isSat){
              expr.children[0].value = expr.children[1].value;
              chooseChildValue(expr.children[0]);
            }
          }
          break;
        }
      case "LTOE":
        {
          if (expr.children[0].value <= expr.children[1].value) {
            return true;
          } else if (expr.children[1].isSat) {
            expr.children[0].value = expr.children[1].value;
            chooseChildValue(expr.children[0]);
          } else {
            expr.children[1].value = expr.children[0].value;
          }
          break;
        }
      case "GTOE":
        {
          if (expr.children[0].value >= expr.children[1].value) {
            return true;
          } else if (expr.children[1].isSat) {
            expr.children[0].value = expr.children[1].value;
            chooseChildValue(expr.children[0]);
          } else {
            expr.children[1].value = expr.children[0].value;
          }
          break;
        }
      case "LessThan":
        {
          if (expr.children[0].value < expr.children[1].value) {
            return true;
          } else if (expr.children[1].isSat) {
            expr.children[0].value = expr.children[1].value;
            chooseChildValue(expr.children[0]);
          } else {
            expr.children[1].value = expr.children[0].value;
          }
          break;
        }
      case "GreaterThan":
        {
          if (expr.children[0].value > expr.children[1].value) {
            return true;
          } else if (expr.children[1].isSat) {
            expr.children[0].value = expr.children[1].value;
            chooseChildValue(expr.children[0]);
          } else {
            expr.children[1].value = expr.children[0].value;
          }
          break;
        }
      case "NotEquals":
        {
          if (expr.children[0].value != expr.children[1].value) {
            return true;
          } else {
            chooseChildValue(expr.children[0]);
            break;
          }
        }
        return false;
    }
  }

  bool chooseChildValue(Expression expr){
    if(expr.children.length == 0){
      return true;
    }
    switch(expr.type){
      case "Multiply":{
        chooseChildValue(expr.children[0]);
        chooseChildValue(expr.children[1]);
      }
    }
    return false; //unable to assign values to make expression SAT
  }
}

// elements of the abstract syntax tree of a equation and it's sub-elements
class Expression {
  bool isSat=false;
  String varName;  // @TODO factory for variable name
  num value; // @TODO factory for constant value
  String type; // as defined in Order.list
  Expression parent;
  List<Expression> children;

  Expression(); //default expression constructor
  Expression.parenthesis(Expression parent, Expression inside){
    type = "Parenthesis";
    children.add(inside);
  }
  Expression.equals(Expression parent, Expression exprLeft, Expression exprRight){
    type = "Equals";
    children.add(exprLeft);
    children.add(exprRight);
  }
  Expression.ltoe(Expression parent, Expression less, Expression more){
    type = "LTOE";
    children.add(less);
    children.add(more);
  }
  Expression.gtoe(Expression parent, Expression more, Expression less){
    type = "GTOE";
    children.add(more);
    children.add(less);
  }
  Expression.lessThan(Expression parent, Expression less, Expression more){
    type = "LessThan";
    children.add(less);
    children.add(more);
  }
  Expression.greaterThan(Expression parent, Expression more, Expression less){
    type = "GreaterThan";
    children.add(more);
    children.add(less);
  }
  Expression.notEquals(Expression parent, Expression exprLeft, Expression exprRight){
    type = "NotEquals";
    children.add(exprLeft);
    children.add(exprRight);
  }
  Expression.power(Expression parent, Expression base, Expression power){
    type = "Power";
    children.add(base);
    children.add(power);
  }
  Expression.multiply(Expression parent, Expression exprLeft, Expression exprRight){
    type = "Multiply";
    children.add(exprLeft);
    children.add(exprRight);
  }
  Expression.divide(Expression parent, Expression exprLeft, Expression exprRight){
    type = "Divide";
    children.add(exprLeft);
    children.add(exprRight);
  }
  Expression.add(Expression parent, Expression exprLeft, Expression exprRight){
    type = "Add";
    children.add(exprLeft);
    children.add(exprRight);
  }
  Expression.subtract(Expression parent, Expression exprLeft, Expression exprRight){
    type = "Subtract";
    children.add(exprLeft);
    children.add(exprRight);
  }
  Expression.variable(Expression parent, String varName){
    type = "Variable";
  }
  Expression.constant(Expression parent, String varName){
    type = "Constant";
  }
}

//creates Expression AST's from string's of equations or functions
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
    thisExpr.varName = matchString;
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