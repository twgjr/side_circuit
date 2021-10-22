import 'values.dart';
import 'expression.dart';
import 'model.dart';
import 'valuerange.dart';

class Solver {
  Order order = Order();
  Model model;
  List<Expression> variables = [];
  List<Expression> visited = [];
  List<Expression> backtracks = [];
  Expression head = Expression.empty(null);
  bool isBackTracking = false;

  Solver(this.model);

  bool solve() {
    model.buildRoot();
    model.root!.printTree();

    // generate the list of unvisited variables
    for (Expression item in model.variables) {
      this.variables.add(item);
      item.setupQueue();
    }

    if (this.variables.isEmpty) {
      return true;
    }

    return check();
  }

  bool check() {
    this.head = this.variables.last;

    while (true) {
      if (this.isBackTracking) {
        this.backtrack();
        continue;
      }

      print("visiting ${this.head.toString()}");
      this.visited.add(this.head);

      if (this.head.isVariable()) {
        if (this.head.edges.length <= 1 && this.variables.isNotEmpty) {
          // no more parents to come back and evaluate later, remove it
          this.variables.removeLast();
        }
      }

      // only evaluate expressions with children that have been set
      if (this.head.isNotReady()) {
        print('not ready. ${this.head.children[0].toString()}-edges=${this.head.children[0].edges.length},'
            ' ${this.head.children[1].toString()}-edges=${this.head.children[1].edges.length}');
        //grab next variable to visit
        if (this.variables.isNotEmpty) {
          this.head = this.variables.last;
        }
        continue;
      }

      // all children visited at this point, set variable or evaluate expr
      if (this.head.isAtFirstPass()) {
        // only set the first time
        if (!setValue()) {
          // failed to set value, start backtracking
          this.isBackTracking = true;
          continue;
        }
      }

      // go to the head's parent
      if (this.head.edges.isNotEmpty) {
        print(
            "removing edge ${this.head.edges.last.toString()} of ${this.head.toString()}");
        this.head = this.head.edges.removeLast();
        continue;
      } else {
        // reached the root with sat result
        return true;
      }
    }
  }

  void backtrack() {
    if (this.head.isVariable()) {
      this.backtracks.add(this.head);
      if (this.backtracks.length > this.head.parents.length) {
        // already attempted to revise this variable, backtrack next variable
        this.backtracks.clear();
        this.variables.add(head); // reset this variable
        print("added ${this.head.toString()} to variables");
      } else if (this.backtracks.length == this.head.parents.length) {
        // ready to attempt revise the variable
        this.revise();
        this.variables.add(head); // reset this variable
        this.isBackTracking = false;
        return;
      }
    } else {
      if (this.head.isNotRoot()) {
        this.revise();
      }
    }
    // continue backtracking
    if (this.visited.isNotEmpty) {
      // add current head to edges of next head from visited list
      if (this.visited.last.parents.contains(this.head)) {
        this.visited.last.edges.add(this.head);
        print(
            "adding edge ${this.head.toString()} to ${this.visited.last.toString()}");
      }
      this.head = this.visited.removeLast();
      print("removed from visited: ${this.head.toString()}");
    }
  }

  bool revise() {
    print("revising ${this.head.toString()}");
    if (this.head.isNotVariable() & this.head.parents.isNotEmpty) {
      print("clearing range of ${this.head.type}");
      this.head.range.clear();
    }
    Range insertSatRange = satRange();
    print("revised range is ${insertSatRange.toString()}");
    if (insertSatRange.isNotEmpty) {
      this.head.insert(insertSatRange);
      this.head.printRange();
    } else {
      print("range not revised: range is ${this.head.range.toString()}");
      return false;
    }
    return true;
  }

  bool setValue() {
    if (this.head.isVariable()) {
      if (!this.head.setNearTarget()) {
        return false;
      }
      this.head.printExpr();
      return true;
    } else if (this.head.isNotConstant()) {
      if (evaluate()) {
        print(
            "evaluated ${this.head.toString()} to ${this.head.value.toString()} from ${this.head.range.toString()}");
        return true;
      } else {
        print(
            "failed to evaluate ${this.head.toString()} to ${this.head.value.toString()} from ${this.head.range.toString()}");
        return false;
      }
    }
    assert(false, "reach unknown state");
    return false;
  }

  bool evaluate() {
    switch (this.head.type) {
      case "Add":
        this.head.value =
            this.head.children[0].value + this.head.children[1].value;
        break;
      case "And":
        this.head.value =
            this.head.children[0].value.and(this.head.children[1].value);
        break;
      case "Or":
        this.head.value =
            this.head.children[0].value.or(this.head.children[1].value);
        break;
      case "Equals":
        this.head.value =
            this.head.children[0].value.equals(this.head.children[1].value);
        break;
      case "LessThan":
        this.head.value =
            this.head.children[0].value < this.head.children[1].value;
        break;
      case "GreaterThan":
        this.head.value =
            this.head.children[0].value > this.head.children[1].value;
        break;
      default:
        this.head.value = Value.logic(false);
        break;
    }

    if (this.head.range.contains(this.head.value)) {
      return true;
    } else {
      return false;
    }
  }

  /// return the range of the given expression that satisfies its immediate
  /// parent(s) and sibling(s).  If cannot satisfy any single expression,
  /// return the empty range
  Range satRange() {
    Range satRange = Range.empty();
    for (Expression parent in this.head.parents) {
      Range newRange = singleSatRange(this.head, parent);
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
    if (sibling.isReady()) {
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
