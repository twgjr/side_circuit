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

    // simplify formulas with constants
    for (Expression constant in model.constants) {
      if (!propagateValue(constant)) {
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
    return checkVariable(firstVariable, 0);
  }

  /// checkVariable sets a variable, then propagates that variable up the tree
  /// return false if any variable does not satisfy a formula
  /// return true if variable and following variables resulted in SAT
  bool checkVariable(Expression variable, int varNum) {
    // identify a range that satisfies all parents and siblings
    // if empty range returned, then cannot satisfy, return unsat
    Range range = satRange(variable);
    if(range.isEmpty){
      return false;
    }
    variable.range = range;

    // set the value based on the sat range
    variable.setMid();
    if (propagateValue(variable)) {
      if (checkNextVariable(variable, varNum)) {
        return true;
      }
    }

    // try the min value from the sat range
    variable.setMin();
    if (propagateValue(variable)) {
      if (checkNextVariable(variable, varNum)) {
        return true;
      }
    }

    // finally try the max value from the sat range
    variable.setMax();
    if (propagateValue(variable)) {
      return checkNextVariable(variable, varNum);
    }

    // all attempts failed, return unsat
    return false;
  }

  bool checkNextVariable(Expression variable, int varNum) {
    // variable was set, check next variable
    if (varNum < unvisitedVars.length) {
      // another variable was available, check it
      Expression nextVariable = unvisitedVars[varNum + 1];
      return checkVariable(nextVariable, varNum + 1);
    }
    return true;
  }

  /// assumes expr is already set
  /// false = failure to set a value that was ready to be set (not sat)
  ///         and populate propRange with a range that would satisfy all parents
  /// true = was able to set all expressions that were ready to be set
  ///        or reached an expression that was not ready to be set
  bool propagateValue(Expression expr) {
    for (Expression parent in expr.parents) {
      if (parent.allChildrenAreVisited()) {
        if (evaluateChildren(parent)) {
          // branch up and attempt to propagate the parent
          if (!propagateValue(parent)) {
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

  bool evaluateChildren(Expression expr) {
    print("evaluate");
    var tempVal;
    switch (expr.type) {
      case "Add":
        tempVal = expr.children[0].value.stored + expr.children[1].value.stored;
        break;
      case "And":
        tempVal =
            expr.children[0].value.stored && expr.children[1].value.stored;
        break;
      case "Equals":
        tempVal =
            expr.children[0].value.stored == expr.children[1].value.stored;
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
  Range satRange(Expression variable) {
    List<Range> varRanges = [];
    for (Expression parent in variable.parents) {
      Range newRange = singleSatRange(variable, parent);
      if (newRange.isEmpty) {
        return newRange;
      }
      varRanges.add(newRange);
    }
    return consolidateRanges(varRanges);
  }

  Range consolidateRanges(List<Range> ranges){
    Value lowestUpper = Value.negInf();  // lowest upper as possible
    Value highestLower = Value.posInf();  // highest lower possible
    for(Range range in ranges) {
      // get highest lower
      if(range.lower.isHigherThan(highestLower)){
        highestLower = range.lower;
      }
      // get lowest upper
      if(range.upper.isLowerThan(lowestUpper)){
        lowestUpper = range.upper;
      }
    }

    if(highestLower.isHigherThan(lowestUpper)) {
      return Range.empty();
    }

    if(highestLower.isSameAs(lowestUpper) &&
        highestLower.isExclusive) {
      return Range.empty();
    }

    return Range.boundary(highestLower, lowestUpper);
  }

  /// check check one parent and sibling for a sat range for the variable
  /// return the range if available, return empty range if not.
  Range singleSatRange(
      Expression variable, Expression parent) {
    int sibIndex = variable.siblingIndex(parent);
    Expression sibling = parent.children[sibIndex];
    if (sibling.isVisited) {
      return withSibVal(variable, parent, sibling);
    } else {
      return withNoSibVal(variable, parent, sibling);
    }
  }

  /// variable must have a range that satisfies the sibling VALUE and parent range
  Range withSibVal(Expression variable, Expression parent, Expression sibling){
    Range range = Range.empty();

    var sibVal = sibling.value.stored;
    print("satisfy range");
    switch (parent.type) {
      case "Add":
        {
          range = Range.copyShiftLeftValue(sibVal, parent.range);
          break;
        }
    }
    return range;
  }

  /// variable must have a range that satisfies the sibling RANGE and parent range
  Range withNoSibVal(Expression variable, Expression parent, Expression sibling){
    Range range = Range.empty();

    print("satisfy value");
    switch (parent.type) {
      case "Add":
        {
          range = Range.copyShiftLeftValue(sibVal, parent.range);
          break;
        }
    }
    return range;
  }
}
