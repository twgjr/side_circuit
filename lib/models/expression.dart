import 'values.dart';
import 'model.dart';
import 'valuerange.dart';

class Order {
  List<Map<String, String>> list = [
    {"Parenthesis": "\\(([^)]+)\\)"},
    {"And": "\\(and([^)]+)\\)"},
    {"Equals": "\\=="},
    {"LTOE": "\\<="},
    {"GTOE": "\\>="},
    {"LessThan": "\\<"},
    {"GreaterThan": "\\>"},
    {"NotEquals": "\\!="},
    {"Power": "\\^"},
    {"Multiply": "\\*"},
    {"Divide": "\\/"},
    {"Add": "\\+"},
    {"Subtract": "\\-"},
    {"Variable": "((?=[^\\d])\\w+)"},
    // variable alphanumeric, not numeric alone
    {"Constant": "^[+-]?((\\d+(\\.\\d+)?)|(\\.\\d+))\$"},
  ];
}

/// elements of the abstract syntax tree of a equation and it's sub-elements
class Expression {
  bool isVisited = false;
  String varName = "";
  Value value;
  String type = ""; // as defined in Order.list
  List<Expression> parents = []; // only variables can have multiple parents
  List<Expression> children = [];
  Model? model;
  Range range;
  Value? _target; // set to -inf for minimize, +inf for maximize,
  // any other value with optimize near that point

  Expression(this.model)
      : this.value = Value.empty(),
        this.range = Range.empty();

  Expression.constant(this.model, Value val)
      : this.value = val,
        this.range = Range.empty() {
    this.type = "Constant";
    this._target = val;
    this.isVisited = true;
  }

  Expression.variable(this.model, this.varName)
      : this.value = Value.empty(),
        this.range = Range.empty() {
    this.type = "Variable";
  }

  Expression.and(this.model)
      : this.value = Value.empty(),
        this.range = Range.logic() {
    this.type = "And";
  }

  int siblingIndex(Expression parent) {
    if (parent.children.length > 1) {
      int index = parent.children.indexOf(this);
      if(index == 0){
        return 1;
      } else {
        return 0;
      }
    }
    return -1; // no sibling
  }

  Value? get target {
    if (this._target == null) {
      if (this.isLogic()) {
        //set true
        this._target = Value.logic(true);
      } else {
        this._target = Value.number(0);
      }
    }
    return this._target;
  }

  bool get isNotVisited {
    return !this.isVisited;
  }

  void setMid() {
    this.value.stored = this.range.midVal();
    this.isVisited = true;
  }

  void setMax() {
    this.value.stored = this.range.upper;
    this.isVisited = true;
  }

  void setMin() {
    this.value.stored = this.range.lower;
    this.isVisited = true;
  }

  bool isLogic() {
    return (this.type == "And" ||
        this.type == "Equals" ||
        this.type == "GreaterThan" ||
        this.type == "LessThan" ||
        this.type == "LTOE" ||
        this.type == "GTOE" ||
        ((this.type == "Constant") && this.value.isLogic()));
  }

  bool isConstant() => this.type == "Constant";

  bool isVariable() => this.type == "Variable";

  bool hasLogicChildren() {
    for (Expression child in this.children)
      if (!child.isLogic()) {
        return false;
      }
    return true;
  }

  List<Expression> unvisitedSiblings(){
    List<Expression> siblings = [];
    for(Expression parent in this.parents){
      Expression sibling = parent.children[this.siblingIndex(parent)];
      if(sibling.isNotVisited){
        siblings.add(sibling);
      }
    }
    return siblings;
  }

  List<Expression> visitedSiblings(){
    List<Expression> siblings = [];
    for(Expression parent in this.parents){
      Expression sibling = parent.children[this.siblingIndex(parent)];
      if(sibling.isVisited){
        siblings.add(sibling);
      }
    }
    return siblings;
  }

  bool allChildrenAreVisited(){
    for(Expression child in this.children){
      if(!child.isVisited || (child.value.stored == null)){
        return false;
      }
    }
    return true;
  }

  void setMaxRange(Value newUpper) {
    // only set a new max if it is less than existing max(s)
    // therefore shrinks the range.  Ignore otherwise
    //  @todo return with error if it ends up with a range of size 0 (no solution possible)
    range.setUpper(newUpper);
  }

  void setMinRange(Value newLower) {
    // only set a new min if it is more than existing min(s)
    // therefore shrinks the range.  Ignore otherwise
    //  @todo return with error if it ends up with a range of size 0 (no solution possible)
    range.setLower(newLower);
  }

  /// Find which sibling number an expression has in a proxy parent expression
  /// Requires diagram proxy parent to guarantee not traversing up
  /// a wrong parent branch from a variable.
  int breadth(Expression parent) {
    return parent.children.indexOf(this);
  }

  /// Find how many levels down the AST an expression lives
  /// If there is more than one parent, it returns max path length
  int depth() {
    int maxDepth = 0;
    int depth = 0;

    if (this.parents.isEmpty) {
      return 0; // already at the root
    }

    depth += 1;
    for (Expression parent in this.parents) {
      while (parent.parents.isNotEmpty) {
        parent = parent.parents[0];
        depth += 1;
      }

      if (depth > maxDepth) {
        maxDepth = depth;
      }
    }
    return maxDepth;
  }

  /// Find which parent expression has this one as child
  /// Requires diagram proxy parent to guarantee not traversing up
  /// a wrong parent branch from a variable.
  int parentNumber(Expression parent) {
    return this.parents.indexOf(parent);
  }

  void printTree() {
    //print the first expression
    String spacer = "";
    for (int ctr = 0; ctr < this.depth(); ctr++) {
      spacer += "->";
    }
    print("$spacer${this.depth()},"
        "0,"
        "${this.type},"
        "${this.varName},"
        "${this.model!.variables.indexOf(this)}");
    for (Expression child in this.children) {
      continuePrintTree(child);
    }
  }

  void continuePrintTree(Expression expr) {
    //breadth first print children
    String spacer = "";
    for (int ctr = 0; ctr < expr.depth(); ctr++) {
      spacer += "->";
    }
    print("$spacer${expr.depth()},"
        "${expr.breadth(expr.parents[0])},"
        "${expr.type},"
        "${expr.varName},"
        "${this.model!.variables.indexOf(expr)}");
    for (Expression child in expr.children) {
      continuePrintTree(child);
    }
  }
}
