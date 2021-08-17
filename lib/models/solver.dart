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
    model.buildRoot();
    model.root!.printTree();

    // // simplify formulas with constants
    // for (Expression constant in model.constants) {
    //   if (!propagate(constant)) {
    //     return false;
    //   }
    // }

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
    // save the state of the range in case need to reset after trying all
    // ranges.
    Range saveRange = firstVariable.range;
    if(check(firstVariable, 0)){
      return true;
    } else {
      firstVariable.range = saveRange;
      firstVariable.isVisited = false;
      return false;
    }
  }

  /// checkVariable sets a variable, then propagates that variable up the tree
  /// return false if any variable does not satisfy a formula
  /// return true if variable and following variables resulted in SAT
  bool check(Expression variable, int varNum) {
    print(
        "checking variable = ${variable.varName}, isLogic = ${variable.valueIsLogic()}");

    if (variable.valueIsLogic()) {
      // variable is logic, try true then false
      variable.value.stored = true;
      variable.isVisited = true;
      print("set logic variable to ${variable.value.stored}");
      if (propagate(variable, varNum)) {
          return true;
      }

      variable.value.stored = false;
      variable.isVisited = true;
      print("set logic variable to ${variable.value.stored}");
      if (propagate(variable, varNum)) {
          return true;
      }
    } else {
      // identify a range that satisfies all parents and siblings
      // if empty range returned, then cannot satisfy, return unsat
      Range range = satRange(variable);
      if (range.isEmpty) {
        return false;
      }
      for (Value value in range.values) {
        variable.range.insert(value);
      }

      List<Range> validPairs = variable.range.validRanges();

      for (Range pair in validPairs) {
        variable.range = pair;
        variable.printRange();

        // set the value based on the sat range
        variable.setNearTarget();
        print("set variable nearest target ${variable.value.stored}");
        if (propagate(variable, varNum)) {
          return true;
        }

        variable.setMin();
        print("set variable to min = ${variable.value.stored}");
        if (propagate(variable, varNum)) {
          return true;
        }

        variable.setMax();
        print("set variable nearest target ${variable.value.stored}");
        if (propagate(variable, varNum)) {
          return true;
        }
      }
    }

    // all attempts failed, return unsat
    return false;
  }

  bool checkNext(int nextVarNum) {
    // variable was set, check next variable
    if ( nextVarNum < unvisitedVars.length) {
      // another variable was available, check it
      Expression nextVariable = unvisitedVars[nextVarNum];
      Range saveRange = nextVariable.range;
      if(check(nextVariable, nextVarNum)) {
        return true;
      } else {
        nextVariable.range = saveRange;
        nextVariable.isVisited = false;
        return false;
      }
    }
    return true;
  }

  /// assumes expr is already set
  /// false = failure to set a value that was ready to be set (not sat)
  ///         and populate propRange with a range that would satisfy all parents
  /// true = was able to set all expressions that were ready to be set
  ///        or reached an expression that was not ready to be set
  bool propagate(Expression expr, int currentVarNum) {
    // update sat ranges of siblings of expr after setting expr value
    // save any ranges before updating for later resetting if unsat
    List<Range> saveRanges = [];
    List<Expression> siblingsVisited = [];
    for (Expression parent in expr.parents) {
      for (Expression child in parent.children) {
        if (child.isNotVisited) {
          saveRanges.add(child.range);
          siblingsVisited.add(child);

          Range range = satRange(child);
          if (range.isEmpty) {
            assert(false,"invalid empty range");
          }
          for (Value value in range.values) {
            child.range.insert(value);
          }
        }
      }
    }

    // check if any parents can be evaluated after setting expr value
    for (Expression parent in expr.parents) {
      if (parent.allChildrenAreVisited()) {
        if (evaluateChildren(parent)) {
          if (parent.isVisited) {
            // branch up and attempt to propagate the parent
            if (!propagate(parent,currentVarNum)) {
              parent.isVisited = false;
              // reset visited children ranges
              for (int i = 0; i < siblingsVisited.length; i++) {
                siblingsVisited[i].range = saveRanges[i];
                siblingsVisited[i].isVisited = false;
              }
              return false;
            }
          }
        } else {
          // could not set parent with given children, expr value is bad
          expr.isVisited = false; // reset visited label
          // reset visited children ranges
          for (int i = 0; i < siblingsVisited.length; i++) {
            siblingsVisited[i].range = saveRanges[i];
            siblingsVisited[i].isVisited = false;
          }
          return false;
        }
      }
    }

    // reaching this point means all possible propagation was either sat or
    // terminated well to the point of checking the next variable.
    if (checkNext(currentVarNum+1)) {
      return true;
    } else {
      // reset the expr that is being propagated in this scope, then continue
      // backtracking for each past call to propagate, restoring ranges
      expr.isVisited = false;
      // reset visited sibling ranges
      for (int i = 0; i < siblingsVisited.length; i++) {
        siblingsVisited[i].range = saveRanges[i];
        siblingsVisited[i].isVisited = false;
      }
      return false;
    }
  }

  bool evaluateChildren(Expression expr) {
    Value tempVal;
    switch (expr.type) {
      case "Add":
        tempVal = expr.children[0].value + expr.children[1].value;
        break;
      case "And":
        tempVal = expr.children[0].value.and(expr.children[1].value);
        break;
      case "Or":
        tempVal = expr.children[0].value.or(expr.children[1].value);
        break;
      case "Equals":
        tempVal = expr.children[0].value.equals(expr.children[1].value);
        break;
      case "LessThan":
        tempVal = expr.children[0].value < expr.children[1].value;
        break;
      case "GreaterThan":
        tempVal = expr.children[0].value > expr.children[1].value;
        break;
      default:
        tempVal = Value.logic(false);
        break;
    }

    if (tempVal.isLogic()) {}

    if (expr.range.contains(tempVal)) {
      expr.isVisited = true;
      expr.value = tempVal;
      return true;
    } else {
      return false;
    }
  }

  /// return the range of the given variable that satisfies its immediate
  /// parent(s) and sibling(s).  If cannot satisfy any single expression,
  /// return the empty range
  Range satRange(Expression variable) {
    Range satRange = Range.empty();
    for (Expression parent in variable.parents) {
      Range newRange = singleSatRange(variable, parent);
      if (newRange.isEmpty) {
        return newRange;
      }
      for (Value value in newRange.values) {
        satRange.insert(value);
      }
    }
    return satRange;
  }

  /// check check one parent and sibling for a sat range for the variable
  /// return the range if available, return empty range if not.
  Range singleSatRange(Expression variable, Expression parent) {
    int sibIndex = variable.siblingIndex(parent);
    Expression sibling = parent.children[sibIndex];
    if (sibling.isVisited) {
      return withSibVal(variable, parent, sibling);
    } else {
      return withNoSibVal(variable, parent, sibling);
    }
  }

  /// variable must have a range that satisfies the sibling VALUE and parent range
  Range withSibVal(Expression variable, Expression parent, Expression sibling) {
    var sibVal = sibling.value.stored;
    switch (parent.type) {
      case "Equals":
        return Range.singleNum(sibVal);
      case "Add":
        return Range.shiftLeft(sibVal, parent.range);
      case "LessThan":
        {
          if (variable.isLeftChildOf(parent)) {
            return Range.upperBoundNum(sibVal, true);
          } else {
            return Range.lowerBoundNum(sibVal, true);
          }
        }
      case "GreaterThan":
        {
          if (variable.isLeftChildOf(parent)) {
            return Range.lowerBoundNum(sibVal, true);
          } else {
            return Range.upperBoundNum(sibVal, true);
          }
        }
    }
    return Range.empty();
  }

  /// variable must have a range that satisfies the sibling RANGE and parent range
  Range withNoSibVal(
      Expression variable, Expression parent, Expression sibling) {
    switch (parent.type) {
      case "Equals":
        return Range.copy(sibling.range);
      case "Add":
        return Range.satisfyAdd(variable.range, parent.range, sibling.range);
      case "LessThan":
        {
          if (variable.isLeftChildOf(parent)) {
            return Range.upperBoundNum(sibling.range.highest.stored, true);
          } else {
            return Range.lowerBoundNum(sibling.range.lowest.stored, true);
          }
        }
      case "GreaterThan":
        {
          if (variable.isLeftChildOf(parent)) {
            return Range.lowerBoundNum(sibling.range.lowest.stored, true);
          } else {
            return Range.upperBoundNum(sibling.range.highest.stored, true);
          }
        }
    }
    return Range.empty();
  }
}
