import 'package:side_circuit/models/values.dart';

import 'expression.dart';
import 'model.dart';
import 'valuerange.dart';

enum SolverState { Unknown, Sat, Unsat }

class Solver {
  Order order = Order();
  Model model;

  Solver(this.model);

  bool solve() {
    model.buildRoot(); // @todo create getter for root that autobuilds
    return check(model.root, null);
  }

  /// starts with the root, null range defaults to target value
  /// branches to children and splits ranges to branch
  /// starts at target then works outward, doubling the range
  /// Example progression of range: target is 0...
  /// {[0,0]} -> {[-1,0),(0,1]} -> {[-2,-1),(1,2]} -> {[-4,-2),(2,4]}...
  /// ...-> {[-inf,-inf/2),(inf/2,inf]}
  bool check(Expression expr, ValueRange range) {
    // do nothing for constants
    if (expr.isConstant()) {
      print("expr ${expr.type} value set to ${expr.value.value}");
      return true;
    }

    if (range == null) {
      range = ValueRange.startingTarget(expr.target);
    }

    // branch down each child of expr
    for (Expression child in expr.children) {
      if (!check(child, null)) {
        return false;
      }
    }

    // reaches here first at leaves (constants and variables)
    // then continues back up the tree setting parents
    if (expr.setValue(range)) {
      print("expr ${expr.type} value set to ${expr.value.value}");
      return true;
    } else {
      print("expr ${expr.type} value not set with range = {${range.lower.value.value},${range.upper.value.value}}");
      print("range width = ${range.rangeWidth()}");
      if (range.upper.value.value is num) {
        num rangeWidth = range.rangeWidth();
        if (rangeWidth == 0) {
          rangeWidth += 1;
        }
        if (rangeWidth < Values.posInf().value) {
          ValueRange checkUpper = ValueRange(Boundary.includes(expr.target),
              Boundary.includes(Values.number(rangeWidth)));
          print("upper num check range = {${checkUpper.lower.value.value},${checkUpper.upper.value.value}}");
          check(expr, checkUpper);

          ValueRange checkLower = ValueRange(
              Boundary.includes(Values.number(-rangeWidth)),
              Boundary.includes(expr.target));
          print("lower num check range = {${checkLower.lower.value.value},${checkLower.upper.value.value}}");
          check(expr, checkLower);
        }
      }
      if (range.upper.value.value is bool) {
        if (range.upper.value.value == true) {
          ValueRange checkUpper = ValueRange(range.lower, range.lower);
          print("upper logic check range = {${checkUpper.lower.value.value},${checkUpper.upper.value.value}}");
          check(expr, checkUpper);
        }
        if (Values.negInf().value == false) {
          ValueRange checkLower = ValueRange(range.upper, range.upper);
          print("lower logic check range = {${checkLower.lower.value.value},${checkLower.upper.value.value}}");
          check(expr, checkLower);
        }
      }
    }
    return false;
  }
}
