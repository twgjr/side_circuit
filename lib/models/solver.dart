import 'package:side_circuit/models/values.dart';

import 'expression.dart';
import 'model.dart';
import 'valuerange.dart';

class Solver {
  Order order = Order();
  Model model;
  List<Expression> unvisitedVars = [];

  Solver(this.model);

  bool solve() {
    model.buildRoot(); // @todo create getter for root that auto builds
    model.root!.printTree();
    simplify();
    for(Expression variable in model.variables){
      if(!variable.isVisited) {
        unvisitedVars.add(variable);
      }
    }
    if(unvisitedVars.isEmpty){
      return true;
    }
    Expression firstVariable = unvisitedVars[0];
    ValueRange checkRange = ValueRange.copy(firstVariable.range);
    return check(firstVariable, 0, checkRange);
  }

  void simplify() {
    for (Expression constant in model.constants) {
      propagate(constant);
    }
  }

  bool check(Expression variable, int varNum, ValueRange range) {
    // check midpoint, skip this if it's logic
    if (!variable.isLogic()) {
      setVarMid(variable, range);
      if (propagate(variable)) {
        // check the next variable
        if ((varNum + 1) < unvisitedVars.length) {
          ValueRange checkRange =
              ValueRange.copy(unvisitedVars[varNum + 1].range);
          return check(unvisitedVars[varNum + 1], varNum + 1, checkRange);
        } else {
          return true;
        }
      }
    }
    // check max (or true if logic)
    setVarMax(variable, range);
    if (propagate(variable)) {
      // check the next variable
      if ((varNum + 1) < unvisitedVars.length) {
        ValueRange checkRange =
            ValueRange.copy(unvisitedVars[varNum + 1].range);
        return check(unvisitedVars[varNum + 1], varNum + 1, checkRange);
      } else {
        return true;
      }
    }
    // check min (or false if logic)
    setVarMin(variable, range);
    if (propagate(variable)) {
      // check the next variable
      if ((varNum + 1) < unvisitedVars.length) {
        ValueRange checkRange =
            ValueRange.copy(unvisitedVars[varNum + 1].range);
        return check(unvisitedVars[varNum + 1], varNum + 1, checkRange);
      } else {
        return true;
      }
    }

    if (!variable.isLogic()) {
      ValueRange splitLeft = ValueRange.splitLeft(range);
      if (check(variable, varNum, splitLeft)) {
        return true;
      }

      ValueRange splitRight = ValueRange.splitRight(range);
      if (check(variable, varNum, splitRight)) {
        return true;
      }
    }

    return false;
  }

  /// false = failure to set a value that was ready to be set (not sat)
  /// true = was able to set all expressions that were ready to be set
  ///        or reached an expression that was not ready to be set
  bool propagate(Expression expr) {
    List<Expression> unvisitedChildren = [];
    for (Expression child in expr.children) {
      if (!child.isVisited) {
        unvisitedChildren.add(child);
      }
    }

    if (unvisitedChildren.length == 0) {
      // expression ready to be set from children
      if (!expr.isVariable() && !expr.isConstant()) {
        if (!setExpression(expr)) {
          return false;
        }
      }
    }

    if (unvisitedChildren.length == 1) {
      // set child such that it satisfies the expr and other child
      if (expr.isLogic() || expr.isVisited) {
        setExprFor(
            unvisitedChildren[0], expr.children.indexOf(unvisitedChildren[0]));
      }
    }
    if (expr.parents.isNotEmpty) {
      if (!propagate(expr.parents[0])) {
        return false;
      }
    }
    return true;
  }

  bool setVarMin(Expression variable, ValueRange range) {
    var tempVal;
    if (variable.isLogic()) {
      tempVal = false;
    } else {
      tempVal = range.min().value.value;
      variable.value = Values.dynamic(tempVal);
    }
    print("set min, ${variable.varName} = ${variable.value.value}");
    if (variable.range.contains(tempVal)) {
      variable.isVisited = true;
      return true;
    } else {
      return false;
    }
  }

  bool setVarMax(Expression variable, ValueRange range) {
    var tempVal;
    if (variable.isLogic()) {
      tempVal = true;
    } else {
      tempVal = range.max().value.value;
      variable.value = Values.dynamic(tempVal);
    }
    print("set max, ${variable.varName} = ${variable.value.value}");
    if (variable.range.contains(tempVal)) {
      variable.isVisited = true;
      return true;
    } else {
      return false;
    }
  }

  bool setVarMid(Expression variable, ValueRange range) {
    var tempVal = range.mid().value.value;
    variable.value = Values.dynamic(tempVal);
    print("set mid, ${variable.varName} = ${variable.value.value}");
    if (variable.range.contains(tempVal)) {
      variable.isVisited = true;
      return true;
    } else {
      return false;
    }
  }

  bool setExpression(Expression expr) {
    var tempVal;
    switch (expr.type) {
      case "Add":
        tempVal = expr.children[0].value.value + expr.children[1].value.value;
        break;
      case "And":
        tempVal = expr.children[0].value.value && expr.children[1].value.value;
        break;
      case "Equals":
        tempVal = expr.children[0].value.value == expr.children[1].value.value;
        break;
      case "LessThan":
        tempVal = expr.children[0].value.value < expr.children[1].value.value;
        break;
      case "GreaterThan":
        tempVal = expr.children[0].value.value > expr.children[1].value.value;
        break;
      default:
        tempVal = Values.logic(false).value;
        break;
    }
    expr.value = Values.dynamic(tempVal);
    print("${expr.type} = ${expr.value.value}");
    if (expr.range.contains(tempVal)) {
      expr.isVisited = true;
      return true;
    } else {
      return false;
    }
  }

  /// set value of expr such that it satisfies the parent or the parent + other children
  bool setExprFor(Expression expr, int index) {
    var tempVal;
    switch (expr.parents[0].type) {
      case "Add":
        if (index == 0) {
          tempVal = expr.parents[0].value.value + expr.children[1].value.value;
        } else {
          tempVal = expr.parents[0].value.value + expr.children[0].value.value;
        }
        break;
      case "And":
        tempVal = true;
        break;
      case "Equals":
        if (index == 0) {
          tempVal = expr.parents[0].children[1].value.value;
        } else {
          tempVal = expr.parents[0].children[0].value.value;
        }
        break;
      case "LessThan":
        if (index == 0) {
          if (expr.range
              .contains(expr.parents[0].children[1].value.value - 1)) {
            tempVal = expr.parents[0].children[1].value.value - 1;
          } else {
            expr.range.setUpper(Boundary.excludes(
                Values.copy(expr.parents[0].children[1].value)));
            return true;
          }
        } else {
          if (expr.range
              .contains(expr.parents[0].children[0].value.value + 1)) {
            tempVal = expr.parents[0].children[0].value.value + 1;
          } else {
            expr.range.setLower(Boundary.excludes(
                Values.copy(expr.parents[0].children[0].value)));
            return true;
          }
        }
        break;
      case "GreaterThan":
        if (index == 0) {
          if (expr.range
              .contains(expr.parents[0].children[0].value.value + 1)) {
            tempVal = expr.parents[0].children[0].value.value + 1;
          } else {
            expr.range.setLower(Boundary.excludes(
                Values.copy(expr.parents[0].children[0].value)));
            return true;
          }
        } else {
          if (expr.range
              .contains(expr.parents[0].children[1].value.value - 1)) {
            tempVal = expr.parents[0].children[1].value.value - 1;
          } else {
            expr.range.setUpper(Boundary.excludes(
                Values.copy(expr.parents[0].children[1].value)));
            return true;
          }
        }
        break;
      default:
        tempVal = Values.logic(false).value;
        break;
    }
    expr.value = Values.dynamic(tempVal);
    print("${expr.type} = ${expr.value.value}");
    if (expr.range.contains(tempVal)) {
      expr.isVisited = true;
      return true;
    } else {
      return false;
    }
  }
}
