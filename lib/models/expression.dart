import 'values.dart';
import 'model.dart';
import 'valuerange.dart';

class Order {
  List<Map<String, String>> list = [
    {"Parenthesis": "\\(([^)]+)\\)"},
    {"And": "\\&\\&"},
    {"Or": "\\|\\|"},
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
  List<Expression> edges = [];
  String varName = "";
  Value value;
  String type = ""; // as defined in Order.list
  List<Expression> parents = []; // only variables can have multiple parents
  List<Expression> children = [];
  Model? model;
  Range range;
  Value? _target; // set to -inf for minimize, +inf for maximize,
  // any other value with optimize near that point

  Expression.empty(this.model)
      : this.value = Value.empty(),
        this.range = Range.empty();

  Expression.constant(this.model, Value val)
      : this.value = val,
        this.range = Range.empty() {
    this.type = "Constant";
    this._target = val;
    if (this.value.isLogic()) {
      this.range = Range.singleLogic(val.stored);
    } else {
      this.range = Range.singleNum(val.stored);
    }
  }

  Expression.variable(this.model, this.varName)
      : this.value = Value.empty(),
        this.range = Range.empty() {
    this.type = "Variable";
  }

  Expression.logic(this.model, this.type)
      : this.value = Value.empty(),
        this.range = Range.initLogic();

  Expression.number(this.model, this.type)
      : this.value = Value.empty(),
        this.range = Range.initNumber();

  Expression.solverAnd(this.model)
      : this.value = Value.empty(),
        this.range = Range.singleLogic(true) {
    this.type = "And";
  }

  bool isRoot(){
    return this.parents.length == 0;
  }
  bool isNotRoot(){
    return !this.isRoot();
  }

  bool isAtFirstPass() {
    return this.edges.length == this.parents.length;
  }

  bool isReady() {
    for(Expression child in this.children) {
      for(Expression parent in child.edges) {
        if (parent == this) {
          return false;
        }
      }
    }
    return true;
  }

  bool isNotReady(){
    return !this.isReady();
  }

  // void resetEdges() {
  //   if (this.edges.isEmpty) {
  //     for (Expression parent in this.parents) {
  //       this.edges.add(parent);
  //     }
  //   }
  // }

  void setupQueue() {
    if (this.edges.isEmpty) {
      for (Expression parent in this.parents) {
        this.edges.add(parent);
        print(
            "added ${parent.toString()} to parent queue of ${this.toString()}");
        parent.setupQueue();
      }
    }
  }

  /// find index of the sibling of this expression under the given parent
  int siblingIndex(Expression parent) {
    if (parent.children.length > 1) {
      int index = parent.children.indexOf(this);
      if (index == 0) {
        return 1;
      } else {
        return 0;
      }
    }
    return -1; // no sibling
  }

  /// tell if this expression is the left child of the given parent
  bool isLeftChildOf(Expression parent) {
    return parent.children.indexOf(this) == 0;
  }

  /// tell if this expression is the right child of the given parent
  bool isRightChildOf(Expression parent) {
    assert(parent.children.length == 2);
    return parent.children.indexOf(this) == 1;
  }

  Value? get target {
    if (this._target == null) {
      if (this.valueIsLogic()) {
        //set true
        this._target = Value.logic(true);
      } else {
        this._target = Value.number(0);
      }
    }
    return this._target;
  }

  void insert(Range insertRange) {
    for (Value bound in insertRange.values) {
      this.range.insert(bound);
    }
  }

  /// find value nearest target, return if value possible, false if not
  bool setNearTarget() {
    List<Range> validRanges = this.range.validRanges();
    if (validRanges.isEmpty) {
      return false;
    }
    Range validRange = validRanges.first;
    if (this.valueIsLogic()) {
      if (validRange.contains(this.target!)) {
        this.value = this.target!;
      }
    }

    // choose valid range pair closest to target
    if (validRange.contains(this.target!)) {
      this.value = this.target!;
    } else if (this.target!.stored == validRange.lowest.stored) {
      this.setMin(validRange);
    } else if (this.target!.stored == validRange.highest.stored) {
      this.setMax(validRange);
    } else if (this.target!.stored > validRange.highest.stored) {
      this.setMax(validRange);
    } else {
      this.setMin(validRange);
    }

    return true;
  }

  void setMax(Range validRange) {
    if (validRange.highest.isNotExclusive) {
      this.value.stored = validRange.highest.stored;
      return;
    }

    if (validRange.width() > 1) {
      this.value.stored = validRange.highest.stored - 1;
    } else {
      this.value.stored = validRange.midVal();
    }
  }

  void setMin(Range validRange) {
    if (validRange.lowest.isNotExclusive) {
      this.value.stored = validRange.lowest.stored;
      return;
    }

    if (validRange.width() > 1) {
      this.value.stored = validRange.lowest.stored + 1;
    } else {
      this.value.stored = validRange.midVal();
    }
  }

  bool isEmpty() {
    return this.range.isEmpty;
  }

  bool isNotEmpty() {
    return !this.isEmpty();
  }

  bool valueIsLogic() {
    return this.type == "And" ||
        this.type == "Or" ||
        ((this.type == "Constant") && this.value.isLogic()) ||
        this.parents.every((element) => element.argIsLogic()) ||
        isComparison();
  }

  bool argIsLogic() {
    return this.type == "And" ||
        this.type == "Or" ||
        ((this.type == "Constant") && this.value.isLogic());
  }

  bool isComparison() {
    return this.type == "Equals" ||
        this.type == "GreaterThan" ||
        this.type == "LessThan" ||
        this.type == "LTOE" ||
        this.type == "GTOE";
  }

  bool isBracket() {
    return this.type == "Parenthesis";
  }

  bool isConstant() => this.type == "Constant";

  bool isNotConstant() => this.type != "Constant";

  bool isVariable() => this.type == "Variable";

  bool isNotVariable() => !this.isVariable();

  /// looks at surrounding expression to determine whether a variable is a logic
  /// or a number kind.  Default to number kind if cannot determine.
  void setInitialRange() {
    assert(this.isVariable(), "only setInitialRange() with variables");
    for (Expression parent in this.parents) {
      if (parent.argIsLogic()) {
        this.range = Range.initLogic();
        return; //found a parent to set the kind
      }
    }

    List<Expression> visited = [];
    List<Expression> siblings = this.siblings();
    for (Expression sibling in siblings) {
      visited.add(sibling);
      if (anySiblingIsLogic(this, sibling, visited)) {
        return; //found a sibling to set the kind
      }
    }
    this.range = Range.initNumber();
  }

  bool anySiblingIsLogic(
      Expression variable, Expression sibling, List<Expression> visited) {
    if (sibling.valueIsLogic()) {
      variable.range = Range.initLogic();
      return true;
    }

    List<Expression> nextSiblings = sibling.siblings();
    for (Expression nextSibling in nextSiblings) {
      if (visited.contains(nextSibling)) {
        continue;
      }
      visited.add(nextSibling);
      if (anySiblingIsLogic(variable, nextSibling, visited)) {
        return true;
      }
    }
    return false;
  }

  bool childrenHaveLogicValue() {
    for (Expression child in this.children)
      if (!child.valueIsLogic()) {
        return false;
      }
    return true;
  }

  List<Expression> siblings() {
    List<Expression> siblings = [];
    for (Expression parent in this.parents) {
      Expression sibling = parent.children[this.siblingIndex(parent)];
      siblings.add(sibling);
    }
    return siblings;
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
        "${this.model!.variables.indexOf(this)},"
        "${this.range.toString()}");
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
        "${this.model!.variables.indexOf(expr)},"
        "${this.range.toString()}");
    for (Expression child in expr.children) {
      continuePrintTree(child);
    }
  }

  String toString() {
    switch (this.type) {
      case "And":
        return "&&";
      case "Or":
        return "||";
      case "Equals":
        return "==";
      case "LTOE":
        return "<=";
      case "GTOE":
        return ">=";
      case "LessThan":
        return "<";
      case "GreaterThan":
        return ">";
      case "NotEquals":
        return "!=";
      case "Power":
        return "^";
      case "Multiply":
        return "*";
      case "Divide":
        return "/";
      case "Add":
        return "+";
      case "Subtract":
        return "-";
      case "Variable":
        return varName;
      case "Constant":
        return value.stored.toString();
    }
    return "";
  }

  void printRange() {
    print("${this.toString()} range is ${this.range.toString()}");
  }

  void printExpr() {
    print("${this.toString()}, ${this.range.toString()}: ${this.value.stored}");
  }
}
