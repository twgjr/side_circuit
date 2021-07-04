import 'package:side_circuit/models/values.dart';

import 'expression.dart';
import 'model.dart';
import 'valuerange.dart';

enum States { Init, Target, Upper, Lower }

/// State wraps around an expression to give it more properties
/// and methods related to Solver.
class State {
  Expression expr;
  ValueRange solveRange;
  States id = States.Init;

  // the check range always starts at the target and works outward
  // toward the max range of the expression
  State(this.solveRange,this.expr);

  bool update(State state) {
    //consider the case for bool and num
    switch (state.id) {
      case States.Init:
        state.solveRange.setRangeTo(state.expr.target!);
        state.id = States.Target;
        return true;
      case States.Target:
        state.solveRange.setRangeTo(state.solveRange.upper.value);
        state.id = States.Upper;
        return true;
      case States.Upper:
        state.solveRange.setRangeTo(state.solveRange.lower.value);
        state.id = States.Lower;
        return true;
      case States.Lower:
        break;
    }
    return false;
  }
}

class Solver {
  Order order = Order();
  Model model;

  Solver(this.model);

  bool solve() {
    model.buildRoot(); // @todo create getter for root that auto builds
    model.root!.printTree();
    //root always has an AND expression.
    ValueRange stateRange = ValueRange(Boundary.includes(model.root!.target!),
        Boundary.includes(model.root!.target!));
    State rootState = State(stateRange,model.root!);
    return check(rootState);
  }

  bool check(State state) {
    if (state.expr.children.isEmpty) {
      return setValueFor(state);
    }
    for (Expression child in state.expr.children) {
      ValueRange childStateRange = ValueRange(Boundary.includes(child.target!),
          Boundary.includes(child.target!));
      State childState = State(childStateRange,child);
      while (!check(childState)) {
        if (!childState.update(childState)) {
          return false;
        }
      }
    }
    return setValueFor(state);
  }

  bool setValueFor(State state) {
    Values targetValue = state.expr.target!;

    print("target is ${state.expr.target!.value}");
    var tempVal;
    switch (state.expr.type) {
      case "Variable":
        if (state.solveRange.upper.value.value < targetValue.value) {
          tempVal = state.solveRange.min();
        }
        if (targetValue.value < state.solveRange.lower.value.value) {
          tempVal = state.solveRange.max();
        }
        if (state.solveRange.contains(targetValue.value)) {
          tempVal = targetValue.value;
        }
        break;
      case "Add":
        tempVal = state.expr.children[0].value.value +
            state.expr.children[1].value.value;
        break;
      case "And":
        tempVal = state.expr.children[0].value.value &&
            state.expr.children[1].value.value;
        break;
      case "Equals":
        tempVal = state.expr.children[0].value.value ==
            state.expr.children[1].value.value;
        break;
      case "LessThan":
        tempVal = state.expr.children[0].value.value <
            state.expr.children[1].value.value;
        break;
      case "GreaterThan":
        tempVal = state.expr.children[0].value.value >
            state.expr.children[1].value.value;
        break;
      default:
        tempVal = Values.logic(false).value;
        break;
    }
    if (state.expr.range.contains(tempVal)) {
      state.expr.value = Values.dynamic(tempVal);
      return true;
    }
    return false;
  }
}
