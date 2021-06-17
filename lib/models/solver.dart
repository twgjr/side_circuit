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
      {"Constant":"^[+-]?((\\d+(\\.\\d+)?)|(\\.\\d+))\$"},
    ];
  }
}

class Model {
  Order order = Order();
  List<Expression> expressions; // single list of Expression roots for the Model
  List<Expression> variables; //points to all of the variables in the model
  // variable list should be populated when an larger expression is created

  void sortVariables() {
    // sort variables in order of:
    // - most constrained (smallest range of possible solutions)
    // - most occurrences (number of times appearing in equations
  }

  void sortExpressions() {
    // sort expressions (roots) in order of:
    // - containing the mose constrained variables
    // - most occurrences (number of times appearing in equations
  }

  void simplify() {
    // grab low hanging fruit for easy solutions:
    // - solve simple single level expressions such as 'x = 1' or 'y > 2' or 'pi/2 = sin(theta)'
    // - continue up the tree
  }

  int solve() {
    sortVariables();
    sortExpressions();
    simplify();

    bool decided = false;
    var solverState = SolverState.Unknown;
    while (!decided) {
      switch (solverState) {
        case SolverState.Unknown:
          {
            solverState = check() ? SolverState.Sat : SolverState.Unsat;
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

  bool check() {
    expressions.forEach((exprRoot) {
      if (!checkBoolean(exprRoot)) {
        return false;
      }
    });
    return true;
  }

  bool checkBoolean(Expression expr) {
    switch (expr.type) {
      case "Parenthesis":
        return checkBoolean(expr.children[0]);
      case "Equals":
        return expr.children[0].value == expr.children[1].value;
      case "LTOE":
        return expr.children[0].value <= expr.children[1].value;
      case "GTOE":
        return expr.children[0].value >= expr.children[1].value;
      case "LessThan":
        return expr.children[0].value < expr.children[1].value;
      case "GreaterThan":
        return expr.children[0].value > expr.children[1].value;
      case "NotEquals":
        return expr.children[0].value != expr.children[1].value;
    }
    return false;
  }

  // returns the constants and variables and propagates them upward towards
  // the root boolean expression, starts with a variable
  Expression updateVars(Expression expr){
    // branch out with the variable set as the root.
    // variable has parents in an unlimited number of root expressions

    // somehow branch on variable range of values

    return expr;
  }

  // returns the constants and variables and propagates them upward towards
  // the root boolean expression, starts with a root
  Expression updateRoots(Expression expr) {
    switch (expr.type) {
      case "Variable":
        return expr;
      case "Constant":
        return expr;
      case "Multiply":
        expr.value = updateRoots(expr.children[0]).value *
            updateRoots(expr.children[1]).value;
        return expr;
      case "Divide":
        expr.value = updateRoots(expr.children[0]).value /
            updateRoots(expr.children[1]).value;
        return expr;
      case "Add":
        {
          expr.value = updateRoots(expr.children[0]).value +
              updateRoots(expr.children[1]).value;
          return expr;
        }
      case "Subtract":
        {
          expr.value = updateRoots(expr.children[0]).value -
              updateRoots(expr.children[1]).value;
          return expr;
        }
      default:
        return expr;
    }
  }
}

// elements of the abstract syntax tree of a equation and it's sub-elements
class Expression {
  bool isSat=false;
  String varName = "";  // @TODO factory for variable name
  num value = 0; // @TODO factory for constant value
  String type = ""; // as defined in Order.list
  Expression parent;
  List<Expression> children=[];

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

  int breadth() {
    if (this.parent!=null) {
      return this.parent.children.indexOf(this);
    } else {
      return 0;
    }
  }

  int depth() {
    int count = 0;

    if(this.parent==null){
      return count; //at the real root
    }

    Expression nextItem = this.parent;
    count+=1;

    while(nextItem.parent!=null) {
      nextItem = nextItem.parent;
      count+=1;
    }
    return count;
  }
}

//creates Expression AST's from string's of equations or functions
class Parser
{
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

  void assembleSyntaxTree(String equationString, int depth, Expression parentNode)
  {
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