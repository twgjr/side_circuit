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
    print("checking variable = ${variable.varName}, isLogic = ${variable.valueIsLogic()}");

    if(variable.valueIsLogic()){
      // variable is logic, try true then false
      variable.value.stored = true;
      variable.isVisited = true;
      print("set logic variable to ${variable.value.stored}");
      if (propagateValue(variable)) {
        if (checkNextVariable(variable, varNum)) {
          return true;
        }
      }

      variable.value.stored = false;
      variable.isVisited = true;
      print("set logic variable to ${variable.value.stored}");
      if (propagateValue(variable)) {
        if (checkNextVariable(variable, varNum)) {
          return true;
        }
      }
    } else {
      // identify a range that satisfies all parents and siblings
      // if empty range returned, then cannot satisfy, return unsat
      Range range = satRange(variable);
      if (range.isEmpty) {
        return false;
      }
      variable.range = range;

      // set the value based on the sat range
      variable.setMid();
      print("set variable to mid ${variable.value.stored}");
      if (propagateValue(variable)) {
        if (checkNextVariable(variable, varNum)) {
          return true;
        }
      }

      // try the min value from the sat range
      variable.setMin();
      print("set variable to min ${variable.value.stored}");
      if (propagateValue(variable)) {
        if (checkNextVariable(variable, varNum)) {
          return true;
        }
      }

      // finally try the max value from the sat range
      variable.setMax();
      print("set variable to max ${variable.value.stored}");
      if (propagateValue(variable)) {
        return checkNextVariable(variable, varNum);
      }
    }
    // all attempts failed, return unsat
    return false;
  }

  bool checkNextVariable(Expression variable, int varNum) {
    // variable was set, check next variable
    if (varNum+1 < unvisitedVars.length) {
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
    print("propagating expression ${expr.type}");
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
    print("evaluating ${expr.type}: ${expr.children[0].value.stored}, ${expr.children[1].value.stored}");
    var tempVal;
    switch (expr.type) {
      case "Add":
        tempVal = expr.children[0].value.stored + expr.children[1].value.stored;
        break;
      case "And":
        tempVal =
            expr.children[0].value.stored && expr.children[1].value.stored;
        break;
      case "Or":
        tempVal =
            expr.children[0].value.stored || expr.children[1].value.stored;
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

    if(tempVal is bool){
      print("temp val is bool");
      if (tempVal) {
        expr.isVisited = true;
        expr.value = Value.dynamic(tempVal);
        print("${expr.type} = ${expr.value.stored}");
        return true;
      } else {
        return false;
      }
    }

    if (expr.range.contains(tempVal)) {
      expr.isVisited = true;
      expr.value = Value.dynamic(tempVal);
      print("${expr.type} = ${expr.value.stored}");
      return true;
    } else {
      return false;
    }
  }

  /// return the range of the given variable that satisfies its immediate
  /// parent(s) and sibling(s).  If cannot satisfy, return the empty range
  Range satRange(Expression variable) {
    print("getting sat range");
    Range satRange = Range.empty();
    for (Expression parent in variable.parents) {
      Range newRange = singleSatRange(variable, parent);
      if (newRange.isEmpty) {
        return newRange;
      }
    }
    return satRange;
  }

  /// check check one parent and sibling for a sat range for the variable
  /// return the range if available, return empty range if not.
  Range singleSatRange(
      Expression variable, Expression parent) {
    print("getting single sat range for ${parent.type}/${variable.varName}");
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
    switch (parent.type) {
      case "Equals":
        {
          range = Range.singleNum(sibVal);
          break;
        }
      case "Add":
        {
          range = Range.shiftLeft(sibVal, parent.range);
          break;
        }
    }
    print("with sibling value sat range: ${range.lowest.stored},${range.highest.stored}");
    return range;
  }

  /// variable must have a range that satisfies the sibling RANGE and parent range
  Range withNoSibVal(Expression variable, Expression parent, Expression sibling){
    print("getting sat range for from sib range");
    Range range = Range.empty();
    print("satisfy value");
    switch (parent.type) {
      case "Equals":
        {
          range = Range.copy(sibling.range);
          break;
        }
      case "Add":
        {
          range = Range.satisfyAdd(variable.range, parent.range, sibling.range);
          break;
        }
    }
    return range;
  }
}
