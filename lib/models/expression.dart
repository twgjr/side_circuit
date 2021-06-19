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