import 'values.dart';
import 'expression.dart';
import 'model.dart';
import 'valuerange.dart';

class Solver {
  Order order = Order();
  Model model;
  List<Expression> variables = [];
  int index = 0;
  List<Expression> queue = [];

  Solver(this.model);

  bool solve() {
    model.buildRoot();
    //model.root!.printTree();

    // generate the list of unvisited variables
    for (Expression variable in model.variables) {
      if (!variable.isVisited) {
        this.variables.add(variable);
        variable.setupQueue();
      }
    }
    if (this.variables.isEmpty) {
      return true;
    }

    // start checking with the first variable
    Expression firstVariable = this.variables[index];
    if (check(firstVariable)) {
      return true; // SAT
    } else {
      if (reviseAndCheck(firstVariable, firstVariable)) {
        return true; // SAT
      } else {
        firstVariable.isVisited = false;
        return false; // UNSAT
      }
    }
  }

  Expression getNext() {
    // first go back to variables that were already started
    if (this.queue.isNotEmpty) {
      return this.queue.removeLast();
    }
    // if no more pending variables, grab a fresh one from the list
    if ((this.index + 1) < this.variables.length) {
      index += 1;
      return variables[index];
    } else {
      return Expression.empty(this.model);
    }
  }

  /// checkVariable sets a variable, then propagates that variable up the tree
  /// return false if any variable does not satisfy a formula
  /// return true if variable and following variables resulted in SAT
  bool check(Expression expr) {
    print("visiting ${expr.type}");

    // check if an expression can be evaluated
    // if not, go back to the last variable on the queue with unvisited parents
    // if no variables on the unvisited parents queue, then grab the next
    // variable on the unvisited variables queue
    for (Expression child in expr.children) {
      if (child.isNotVisited) {
        Expression nextExpr = this.getNext();
        if (nextExpr.isEmpty()) {
          return true;
        }
        if (check(nextExpr)) {
          return true;
        } else {
          return reviseAndCheck(expr, nextExpr);
        }
      }
    }

    // set value after reaching all children, ignoring constant
    // this evaluates an operator or chooses a variable value
    // only do the first time visiting an expression
    if(expr.parentQueue.length == expr.parents.length) {
      if (!setValue(expr)) {
        return false;
      }
    }

    // add expr to queue if it will have a parent left to visit
    if (expr.parentQueue.length > 1) {
      this.queue.add(expr);
    } else {
      expr.isVisited = true;
    }

    if(expr.parentQueue.isNotEmpty) {
      Expression nextParent = expr.parentQueue.removeLast();
      if (check(nextParent)) {
        return true;
      } else {
        return reviseAndCheck(expr, nextParent);
      }
    }

    Expression nextExpr = this.getNext();
    if (nextExpr.isEmpty()) {
      return true;
    }
    if (check(nextExpr)) {
      return true;
    } else {
      return reviseAndCheck(expr, nextExpr);
    }
  }

  bool reviseAndCheck(Expression expr, Expression next) {
    print("revising ${expr.toString()}");
    if(expr.isNotVariable()) {
      expr.range.clear();
    }
    Range insertSatRange = satRange(expr);
    print("revised range is ${insertSatRange.toString()}");
    if (insertSatRange.isNotEmpty) {
      expr.insert(insertSatRange);
      expr.printRange();
    } else {
      return false;
    }
    if (expr.isVariable()) {
      // only set variables after revising sat range
      if (!setValue(expr)) {
        return false;
      }
    } else {
      // continue backtracking otherwise
      return false;
    }
    if (check(next)) {
      return true;
    } else {
      print("failed to check ${next.toString()}");
      return false;
    }
  }

  bool setValue(Expression expr) {
    if (expr.isVariable()) {
      if (!expr.setNearTarget()) {
        return false;
      }
      expr.printExpr();
      return true;
    } else if (expr.isNotConstant()) {
      if (evaluateChildren(expr)) {
        print(
            "evaluated ${expr.toString()} to ${expr.value.toString()} from ${expr.range.toString()}");
        return true;
      } else {
        print(
            "failed to evaluate ${expr.toString()} to ${expr.value.toString()} from ${expr.range.toString()}");
        return false;
      }
    }
    assert(false,"reach unknown state");
    return false;
  }

  bool evaluateChildren(Expression expr) {
    switch (expr.type) {
      case "Add":
        expr.value = expr.children[0].value + expr.children[1].value;
        break;
      case "And":
        expr.value = expr.children[0].value.and(expr.children[1].value);
        break;
      case "Or":
        expr.value = expr.children[0].value.or(expr.children[1].value);
        break;
      case "Equals":
        expr.value = expr.children[0].value.equals(expr.children[1].value);
        break;
      case "LessThan":
        expr.value = expr.children[0].value < expr.children[1].value;
        break;
      case "GreaterThan":
        expr.value = expr.children[0].value > expr.children[1].value;
        break;
      default:
        expr.value = Value.logic(false);
        break;
    }

    if (expr.range.contains(expr.value)) {
      return true;
    } else {
      return false;
    }
  }

  /// return the range of the given expression that satisfies its immediate
  /// parent(s) and sibling(s).  If cannot satisfy any single expression,
  /// return the empty range
  Range satRange(Expression expr) {
    Range satRange = Range.empty();
    for (Expression parent in expr.parents) {
      Range newRange = singleSatRange(expr, parent);
      if (newRange.isEmpty) {
        return newRange;
      }
      for (Value value in newRange.values) {
        satRange.insert(value);
      }
    }
    return satRange;
  }

  /// check  one parent and sibling for a sat range for the variable
  /// return the range if available, return empty range if not.
  Range singleSatRange(Expression expr, Expression parent) {
    int sibIndex = expr.siblingIndex(parent);
    Expression sibling = parent.children[sibIndex];
    if (sibling.isVisited) {
      return withSibVal(expr, parent, sibling);
    } else {
      return withNoSibVal(expr, parent, sibling);
    }
  }

  /// variable must have a range that satisfies the sibling VALUE and parent range
  Range withSibVal(Expression expr, Expression parent, Expression sibling) {
    var sibVal = sibling.value.stored;
    switch (parent.type) {
      case "And":
        return Range.singleLogic(true);
      case "Equals":
        return Range.singleNum(sibVal);
      case "Add":
        return Range.shiftLeft(sibVal, parent.range);
      case "LessThan":
        {
          if (expr.isLeftChildOf(parent)) {
            return Range.upperBoundNum(sibVal, true);
          } else {
            return Range.lowerBoundNum(sibVal, true);
          }
        }
      case "GreaterThan":
        {
          if (expr.isLeftChildOf(parent)) {
            return Range.lowerBoundNum(sibVal, true);
          } else {
            return Range.upperBoundNum(sibVal, true);
          }
        }
    }
    return Range.empty();
  }

  /// variable must have a range that satisfies the sibling RANGE and parent range
  Range withNoSibVal(Expression expr, Expression parent, Expression sibling) {
    switch (parent.type) {
      case "And":
        return Range.singleLogic(true);
      case "Equals":
        return Range.copy(sibling.range);
      case "Add":
        return Range.satisfyAdd(parent.range, sibling.range);
      case "LessThan":
        {
          if (expr.isLeftChildOf(parent)) {
            return Range.upperBoundNum(sibling.range.highest.stored, true);
          } else {
            return Range.lowerBoundNum(sibling.range.lowest.stored, true);
          }
        }
      case "GreaterThan":
        {
          if (expr.isLeftChildOf(parent)) {
            return Range.lowerBoundNum(sibling.range.lowest.stored, true);
          } else {
            return Range.upperBoundNum(sibling.range.highest.stored, true);
          }
        }
    }
    return Range.empty();
  }
}
