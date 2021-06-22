import 'values.dart';
import 'model.dart';
import 'range.dart';

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
  //bool isSat = false;
  String varName = "";
  Values value;
  int mode = 0;
  String type = ""; // as defined in Order.list
  List<Expression> parents = []; // only variables can have multiple parents
  List<Expression> children = [];
  Model model;
  List<Range> ranges = [];
  bool isVisited = false;

  Expression(this.model) {
    this.value = Values.number(0);
    this.ranges.add(Range(Boundary.includes(Values.negInf()),
        Boundary.includes(Values.posInf())));
  }

  Expression.constant(this.model,Values val) {
    this.type = "Constant";
    this.value = val;
    this.ranges.add(Range(Boundary.includes(val),Boundary.includes(val)));
  }

  Expression.variable(this.model,Values val) {
    this.value = Values.number(0);
    this.ranges.add(Range(Boundary.includes(Values.negInf()),
        Boundary.includes(Values.posInf())));
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

  bool isConstant() => this.type == "Constant";

  void setRangeFor(Expression exprToSetRange) {
    switch (this.type) {
      case "LessThan":
        {
          if (this.children.indexOf(exprToSetRange) == 0) {
            exprToSetRange.setMaxRange(
                Boundary.excludes(Values.number(this.children[1].value.value)));
          } else {
            exprToSetRange.setMinRange(
                Boundary.excludes(Values.number(this.children[0].value.value)));
          }
          break;
        }
      case "GreaterThan":
        {
          if (this.children.indexOf(exprToSetRange) == 0) {
            exprToSetRange.setMinRange(
                Boundary.excludes(Values.number(this.children[0].value.value)));
          } else {
            exprToSetRange.setMaxRange(
                Boundary.excludes(Values.number(this.children[1].value.value)));
          }
          break;
        }
      case "Equals":
        {
          if (this.children.indexOf(exprToSetRange) == 0) {
            exprToSetRange.setMinRange(
                Boundary.includes(Values.number(this.children[1].value.value)));
            exprToSetRange.setMaxRange(
                Boundary.includes(Values.number(this.children[1].value.value)));
          } else {
            exprToSetRange.setMinRange(
                Boundary.includes(Values.number(this.children[0].value.value)));
            exprToSetRange.setMaxRange(
                Boundary.includes(Values.number(this.children[0].value.value)));
          }
          break;
        }
      case "Add":
        {
          if (this.children.indexOf(exprToSetRange) == 0) {
            exprToSetRange.setMinRange(
                Boundary.includes(Values.number(this.children[1].value.value)));
            exprToSetRange.setMaxRange(
                Boundary.includes(Values.number(this.children[1].value.value)));
          } else {
            exprToSetRange.setMinRange(
                Boundary.includes(Values.number(this.children[0].value.value)));
            exprToSetRange.setMaxRange(
                Boundary.includes(Values.number(this.children[0].value.value)));
          }
          break;
        }
    }
  }

  /// sets the value for whatever the children of expression produce
  /// returns true if was possible to set value
  /// returns false if not possible to set value
  bool setValue(){

  }

  bool solveFor(Expression expr){

  }

  void setMaxRange(Boundary newUpper) {
    // only set a new max if it is less than existing max(s)
    // therefore shrinks the range.  Ignore otherwise
    //  @todo return with error if it ends up with a range of size 0 (no solution possible)
    for (Range range in this.ranges) {
      range.setUpper(newUpper);
    }
  }

  void setMinRange(Boundary newLower) {
    // only set a new min if it is more than existing min(s)
    // therefore shrinks the range.  Ignore otherwise
    //  @todo return with error if it ends up with a range of size 0 (no solution possible)
    for (Range range in this.ranges) {
      range.setLower(newLower);
    }
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
