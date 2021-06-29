import 'dart:html';

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
    // do nothing with expressions already set (like a constant)
    if (expr.isSet()) {
      return true;
    }

    if (range == null) {
      range = ValueRange.startingTarget(expr.target);
    }

    // branch down each child of expr
    for (Expression child in expr.children) {
      if( !check(child, null)) {
        return false;
      }
    }

    // reaches here first at leaves (constants and variables)
    // then continues back up the tree setting parents
    if ( expr.setValue(range) ) {
      print("expr ${expr.type} value set");
      return true;
    } else {
      print("expr ${expr.type} value not set");
      if( range.upper.value.value*2 < Values.posInf().value) {
        ValueRange checkUpper = ValueRange(
            range.upper,Boundary.includes(range.upper.value.value * 2));
        check(expr, checkUpper);
      }
      if( Values.negInf().value < range.lower.value.value*2) {
        ValueRange checkLower = ValueRange(
            Boundary.includes(range.lower.value.value * 2), range.lower);
        check(expr, checkLower);
      }
    }
    return false;
  }
}
