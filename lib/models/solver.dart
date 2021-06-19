import 'package:side_circuit/models/model.dart';

import 'expression.dart';
import 'model.dart';


enum SolverState { Unknown, Sat, Unsat }

class Solver {
  Order order = Order();
  Model model;

  Solver(this.model);

  void sortVariables() {
    // sort variables in order of:
    // - most constrained (smallest range of possible solutions)
    // - most occurrences (number of times appearing in equations
  }

  void sortExpressions() {
    // sort expressions (roots) in order of:
    // - containing the mose constrained variables
    // - most occurrences (number of times appearing in equations
  }

  void simplify() {
    // grab low hanging fruit for easy solutions:
    // - solve simple single level expressions such as 'x = 1' or 'y > 2' or 'pi/2 = sin(theta)'
    // - continue up the tree
  }

  int solve() {
    sortVariables();
    sortExpressions();
    simplify();

    bool decided = false;
    var solverState = SolverState.Unknown;
    while (!decided) {
      switch (solverState) {
        case SolverState.Unknown:
          {
            solverState = check() ? SolverState.Sat : SolverState.Unsat;
            break;
          }
        case SolverState.Sat:
          {
            print("Solution is sat.");
            decided = true;
            break;
          }
        case SolverState.Unsat:
          {
            print("Solution is UNSAT!");
            decided = true;
            break;
          }
      }
    }
    return solverState.index;
  }

  bool check() {
    for (Expression exprRoot in model.expressions) {
      if (!checkBoolean(exprRoot)) {
        return false;
      }
    };
    return true;
  }

  bool checkBoolean(Expression expr) {
    switch (expr.type) {
      case "Parenthesis":
        return checkBoolean(expr.children[0]);
      case "Equals":
        return expr.children[0].value == expr.children[1].value;
      case "LTOE":
        return expr.children[0].value <= expr.children[1].value;
      case "GTOE":
        return expr.children[0].value >= expr.children[1].value;
      case "LessThan":
        return expr.children[0].value < expr.children[1].value;
      case "GreaterThan":
        return expr.children[0].value > expr.children[1].value;
      case "NotEquals":
        return expr.children[0].value != expr.children[1].value;
    }
    return false;
  }

// returns the constants and variables and propagates them upward towards
// the root boolean expression, starts with a variable
  Expression updateVars(Expression expr) {
    // branch out with the variable set as the root.
    // variable has parents in an unlimited number of root expressions

    // somehow branch on variable range of values

    return expr;
  }

// returns the constants and variables and propagates them upward towards
// the root boolean expression, starts with a root
  Expression updateRoots(Expression expr) {
    switch (expr.type) {
      case "Variable":
        return expr;
      case "Constant":
        return expr;
      case "Multiply":
        expr.value = updateRoots(expr.children[0]).value *
            updateRoots(expr.children[1]).value;
        return expr;
      case "Divide":
        expr.value = updateRoots(expr.children[0]).value /
            updateRoots(expr.children[1]).value;
        return expr;
      case "Add":
        {
          expr.value = updateRoots(expr.children[0]).value +
              updateRoots(expr.children[1]).value;
          return expr;
        }
      case "Subtract":
        {
          expr.value = updateRoots(expr.children[0]).value -
              updateRoots(expr.children[1]).value;
          return expr;
        }
      default:
        return expr;
    }
  }
}
