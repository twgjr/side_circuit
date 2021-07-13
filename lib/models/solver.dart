import 'values.dart';
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

    // simplify all the constants
    for (Expression constant in model.constants) {
      if (!propagate(constant)) {
        return false;
      }
    }

    // generate the list of unvisited variables
    for (Expression variable in model.variables) {
      if (!variable.isVisited) {
        unvisitedVars.add(variable);
      }
    }
    if (unvisitedVars.isEmpty) {
      return true;
    }

    // begin attempt to set variables to satisfy formulas
    Expression firstVariable = unvisitedVars[0];
    if (!checkSatisfy(firstVariable)) {
      return false;
    }
    ValueRange checkRange = ValueRange.copy(firstVariable.range);
    return check(firstVariable, 0, checkRange);
  }

  /// check takes in a variable, then propagates that variable up the tree
  /// return false if any variable does not satisfy a formula
  /// return true for all variables resulted in SAT
  bool check(Expression variable, int varNum, ValueRange range) {
    if (variable.isVisited) {
      // variable was already satisfied by some parent + sibling so check next
      // variable
      return checkNextVariable(varNum);
    } else {
      // variable still needs to be set
      if (setVar(variable, varNum, range)) {
        // continue to check next variable
        return checkNextVariable(varNum);
      } else {
        return false;
      }
    }
  }

  /// return true if expr was able to satisfy parent+sibling, or if expr
  /// did was not required to satisfy parent+sibling
  /// return false if attempted to satisfy parent+sibling, but failed
  bool checkSatisfy(Expression expr) {
    for (Expression parent in expr.parents) {
      int siblingIndex = expr.siblingIndex(parent);
      if (siblingIndex >= 0) {
        // has sibling, -1 for no siblings
        Expression sibling = parent.children[siblingIndex];
        if ((parent.isLogic() || parent.isVisited) && sibling.isVisited) {
          if (!satisfy(expr, sibling, siblingIndex, parent)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  bool setVar(Expression variable, int varNum, ValueRange range) {
    if (variable.isLogic()) {
      // check true
      setVarMax(variable, range);
      if (propagate(variable)) {
        return true;
      }
      // check false
      setVarMin(variable, range);
      if (propagate(variable)) {
        return true;
      }
    } else {
      // expr is num, check near target value
      setVarTarget(variable, range);
      if (propagate(variable)) {
        return true;
      }
    }
    return false;
  }

  bool checkNextVariable(int varNum) {
    if ((varNum + 1) < unvisitedVars.length) {
      // first check if there are any immediate expressions that can set this
      // variable
      if (!checkSatisfy(unvisitedVars[varNum + 1])) {
        return false;
      }
      ValueRange checkRange = ValueRange.copy(unvisitedVars[varNum + 1].range);
      return check(unvisitedVars[varNum + 1], varNum + 1, checkRange);
    } else {
      return true; // all variables were checked and everything is SAT
    }
  }

  /// false = failure to set a value that was ready to be set (not sat)
  /// true = was able to set all expressions that were ready to be set
  ///        or reached an expression that was not ready to be set
  bool propagate(Expression expr) {
    print("propagate");

    for (Expression parent in expr.parents) {
      // children ready to set parent (e.g. has two constants as args
      if (parent.allChildrenAreVisited()) {
        if (!evaluate(parent)) {
          // could not set parent
          return false;
        }
        // branch up and attempt to propagate the parent
        if (propagate(parent)) {
          // somewhere up the tree a parent was found to be unsat
          return false;
        }
      }

      for(Expression child in parent.children) {
        if(!child.isVisited) {
          if (!checkSatisfy(child)) {
            return false;
          }
        }
      }
    }
    //if it successfully attempted to propagate without an unsat, return true
    return true;
  }

  bool setVarMin(Expression variable, ValueRange range) {
    var tempVal;
    if (variable.isLogic()) {
      variable.value = Values.logic(false);
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
      variable.value = Values.logic(true);
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

  bool setVarTarget(Expression variable, ValueRange range) {
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

  bool evaluate(Expression expr) {
    print("evaluate");
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

  /// set value of expr such that it satisfies the parent + sibling
  bool satisfy(
      Expression exprToSat, Expression sibling, int siblingIndex, Expression parent) {
    print("satisfy");
    var tempVal;
    switch (parent.type) {
      case "Add":
          tempVal = parent.value.value - sibling.value.value;
        break;
      case "And":
        tempVal = true;
        break;
      case "Equals":
          tempVal = sibling.value.value;
        break;
      case "LessThan":
        if (siblingIndex == 0) {
          exprToSat.range.setLower(
              Boundary.excludes(Values.copy(sibling.value)));
          return true;
        } else {
          exprToSat.range.setUpper(
              Boundary.excludes(Values.copy(sibling.value)));
          return true;
        }
      case "GreaterThan":
        if (siblingIndex == 0) {
          exprToSat.range.setUpper(
              Boundary.excludes(Values.copy(sibling.value)));
          return true;
        } else {
          exprToSat.range.setLower(
              Boundary.excludes(Values.copy(sibling.value)));
          return true;
        }
      default:
        tempVal = Values.logic(false).value;
        break;
    }
    exprToSat.value = Values.dynamic(tempVal);
    print("${exprToSat.type} = ${exprToSat.value.value}");
    if (exprToSat.range.contains(tempVal)) {
      exprToSat.isVisited = true;
      return true;
    } else {
      return false;
    }
  }
}
