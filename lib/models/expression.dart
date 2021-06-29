import 'values.dart';
import 'model.dart';
import 'valuerange.dart';

class Order {
  List<Map<String, String>> list;

  Order() {
    list = [
      {"Parenthesis": "\\(([^)]+)\\)"},
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
}

/// elements of the abstract syntax tree of a equation and it's sub-elements
class Expression {
  String varName = "";
  Values value;
  String type = ""; // as defined in Order.list
  List<Expression> parents = []; // only variables can have multiple parents
  List<Expression> children = [];
  Model model;
  ValueRange range;
  Values target; // set to -inf for minimize, +inf for maximize,
  // any other value with optimize near that point

  Expression(this.model) {
    this.range = ValueRange(
        Boundary.includes(Values.negInf()), Boundary.includes(Values.posInf()));
  }

  Expression.constant(this.model, Values val) {
    this.type = "Constant";
    this.value = val;
    this.range = ValueRange(Boundary.includes(val), Boundary.includes(val));
  }

  Expression.and(this.model) {
    this.type = "And";
    this.range = ValueRange(Boundary.logicLow(), Boundary.logicHigh());
  }

  /// returns the sibling of expr that is a constant.
  /// Return expr if no other siblings are constants
  Expression constantChild(Expression expr) {
    for (Expression child in this.children) {
      if (child == expr) {
        continue;
      }
      if (child.isConstant()) {
        return child;
      }
    }
    return expr;
  }

  bool isSet() => this.value != null;

  bool isConstant() => this.type == "Constant";

  bool isVariable() => this.type == "Variable";

  /// sets the value for whatever the children of expression produce
  /// returns true if was possible to set value
  /// returns false if not possible to set value
  bool setValue(ValueRange solverRange) {
    Values targetValue;
    if (target == null) {
      targetValue = Values.number(0);
    } else {
      targetValue = this.target.value;
    }
    var tempVal;
    switch (this.type) {
      case "Variable":
        if (solverRange.upper.value.value < targetValue.value) {
          tempVal = solverRange.max();
        }
        if (targetValue.value < solverRange.lower.value.value) {
          tempVal = solverRange.min();
        }
        if (solverRange.contains(targetValue.value)) {
          tempVal = targetValue.value;
        }
        break;
      case "Add":
        tempVal = this.children[0].value.value + this.children[1].value.value;
        break;
      case "And":
        tempVal = this.children[0].value.value && this.children[1].value.value;
        break;
      case "Equals":
        tempVal = this.children[0].value.value == this.children[1].value.value;
        break;
      case "LessThan":
        tempVal = this.children[0].value.value < this.children[1].value.value;
        break;
      case "GreaterThan":
        tempVal = this.children[0].value.value > this.children[1].value.value;
        break;
    }
    if (range.contains(tempVal)) {
      this.value = Values.dynamic(tempVal);
      return true;
    }
    return false;
  }

  void setMaxRange(Boundary newUpper) {
    // only set a new max if it is less than existing max(s)
    // therefore shrinks the range.  Ignore otherwise
    //  @todo return with error if it ends up with a range of size 0 (no solution possible)
    range.setUpper(newUpper);
  }

  void setMinRange(Boundary newLower) {
    // only set a new min if it is more than existing min(s)
    // therefore shrinks the range.  Ignore otherwise
    //  @todo return with error if it ends up with a range of size 0 (no solution possible)
    range.setLower(newLower);
  }

  /// Find which sibling number an expression has in a proxy parent expression
  /// Requires diagram proxy parent to guarantee not traversing up
  /// a wrong parent branch from a variable.
  int breadth(Expression parent) {
    if (parent != null) {
      return parent.children.indexOf(this);
    } else {
      return 0;
    }
  }

  /// Find how many levels down the AST an expression lives
  /// Requires diagram proxy parent to guarantee not traversing up
  /// a wrong parent branch from a variable.
  int depth(Expression parent) {
    int count = 0;

    if (parent == null) {
      return count; //at the real root
    }

    count += 1;

    while (parent.parents.isNotEmpty) {
      parent = parent.parents[0];
      count += 1;
    }
    return count;
  }

  /// Find which parent expression has
  /// Requires diagram proxy parent to guarantee not traversing up
  /// a wrong parent branch from a variable.
  int parentNumber(Expression parent) {
    if (parent != null) {
      return this.parents.indexOf(parent);
    } else {
      return 0;
    }
  }
}
