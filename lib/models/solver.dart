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
    Range varRange = Range.copy(firstVariable.range);
    return checkVariable(firstVariable, 0, varRange);
  }

  /// check takes in a variable, then propagates that variable up the tree
  /// return false if any variable does not satisfy a formula
  /// return true for all variables resulted in SAT
  bool checkVariable(Expression variable, int varNum, Range varRange) {
    if(decide(variable, varRange)) {
      if(checkNextVariable(variable, varNum)){
        // no more variables left or all were satisfied
        return true;
      } else {
        // one of the next variables was not satisfied, adjust this variable's range
        if( varRange.width() >= 1) {
          // has not reached a limit yet, can keep branching the variable ranges
          if (checkVariable(variable, varNum, Range.splitLeft(varRange))) {
            return true;
          } else {
            return checkVariable(variable, varNum, Range.splitRight(varRange));
          }
        }
        // failure if reached the recursion limit
        return false;
      }
    } else {
      return false;
    }
  }

  bool checkNextVariable(Expression variable, int varNum) {
    // variable was set, check next variable
    if (varNum < unvisitedVars.length) {
      // another variable was available, check it
      Expression nextVariable = unvisitedVars[varNum + 1];
      Range nextVarRange = Range.copy(nextVariable.range);
      return checkVariable(nextVariable, varNum + 1, nextVarRange);
    }
    return true;
  }

  /// assumes expr is already set
  /// false = failure to set a value that was ready to be set (not sat)
  /// true = was able to set all expressions that were ready to be set
  ///        or reached an expression that was not ready to be set
  bool propagate(Expression expr) {
    //print("propagate");
    for (Expression parent in expr.parents) {
      // children ready to set parent (e.g. has two constants as args
      if (parent.allChildrenAreVisited()) {
        if (evaluate(parent)) {
          // branch up and attempt to propagate the parent
          if (!propagate(parent)) {
            // somewhere up the tree a parent was found to be unsat
            return false;
          }
        } else {
          // could not set parent with given children
          expr.isVisited = false; // reset visited label
          return false;
        }
      }
    }
    //if it successfully attempted to propagate without an unsat, return true
    return true;
  }

  bool setVarMid(Expression variable, Range range) {
    var tempVal = range.midVal();
    variable.value = Value.dynamic(tempVal);
    print("set mid, ${variable.varName} = ${variable.value.stored}");
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
        tempVal = expr.children[0].value.stored + expr.children[1].value.stored;
        break;
      case "And":
        tempVal = expr.children[0].value.stored && expr.children[1].value.stored;
        break;
      case "Equals":
        tempVal = expr.children[0].value.stored == expr.children[1].value.stored;
        break;
      case "LessThan":
        tempVal = expr.children[0].value.stored < expr.children[1].value.stored;
        break;
      case "GreaterThan":
        tempVal = expr.children[0].value.stored > expr.children[1].value.stored;
        break;
      default:
        tempVal = Value.logic(false).stored;
        break;
    }
    expr.value = Value.dynamic(tempVal);
    print("${expr.type} = ${expr.value.stored}");
    if (expr.range.contains(tempVal)) {
      expr.isVisited = true;
      return true;
    } else {
      return false;
    }
  }

  /// return the range of the given variable that satisfies its immediate
  /// parent(s) and sibling(s).  If cannot satisfy, return the empty range
  Range satRangeFromMultipleOf(Expression variable, Range checkRange) {
    List<Range> varRanges = [];
    // fill the varRange list for each parent
    for( Expression parent in variable.parents){
      // add the new range to the list of ranges in order of lower
      // remove any overlap, possibly eliminating some ranges

      Range newRange = satRangeFromSingleOf(variable,parent,checkRange);

      // return empty range if any range is empty
      if(newRange.isEmpty) {
        return newRange;
      }

      varRanges.add(newRange);
    }

    // pass the list of ranges into a function that returns one consolidated range



    Range varRange;
    if(varRanges.isEmpty) {
      varRange = Range.empty();
    } else {
      varRange = varRanges[0];
    }
    return varRange;
  }

  /// check check one parent and sibling for a sat range for the variable
  /// return the range if available, return empty range if not.
  Range satRangeFromSingleOf(Expression variable, Expression parent, Range checkRange) {
    int sibIndex = variable.siblingIndex(parent);
    Expression sibling = parent.children[sibIndex];
    if(!sibling.isVisited){
      // this variable is free to decide by it's own range
      return checkRange;
    } else {
      // need to satisfy the sibling
      var sibVal = parent.children[sibIndex].value.stored;
      print("satisfy");
      Range tempRange;
      switch (parent.type) {
        case "Add":
          tempRange = Range.copyShiftLeft(sibVal, parent.range);
          break;
        case "LessThan":

          break;
        default:
          break;
      }
      expr.value = Value.dynamic(tempVal);
      print("${expr.type} = ${expr.value.stored}");
      if (expr.range.contains(tempVal)) {
        expr.isVisited = true;
        return true;
      } else {
        return false;
      }
    }
    return Range.empty();
  }

  bool decide(Expression variable, Range checkRange) {
    Range varSatRange = satRangeFromMultipleOf(variable, checkRange);
    if (varSatRange.isEmpty) {
      return false;
    } else {
      variable.setMid();
      return true;
    }
  }
}
