import 'values.dart';
import 'model.dart';

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
  bool isSat = false;
  String varName = "";
  Values value;
  String type = ""; // as defined in Order.list
  List<Expression> parents = []; // only variables can have multiple parents
  List<Expression> children = [];
  Model model;

  Expression(this.model){
    this.value = Values.number(0);
  } //default expression constructor

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