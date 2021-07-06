import 'package:side_circuit/models/values.dart';

import 'expression.dart';
import 'model.dart';
import 'valuerange.dart';

class Solver {
  Order order = Order();
  Model model;

  Solver(this.model);

  bool solve() {
    model.buildRoot(); // @todo create getter for root that auto builds
    model.root!.printTree();
    return check(model.variables[0], 0);;
  }

  bool check(Expression variable, int varNum) {
    // track expressions higher in the tree that need to be checked
    // if ready to set it's value
    List<Expression> unvisited = [];
    Expression exprPointer = variable;

    // do loop to set variable and propagate up the tree as far as possible
    do {
      print("starting with pointer at ${exprPointer.type}");
      bool hasUnvisitedChildren = false;
      for (Expression child in exprPointer.children) {
        if (!child.isVisited) {
          hasUnvisitedChildren = true;
          print("unvisited child is ${child.type}");
        }
      }
      if (!hasUnvisitedChildren) {
        setValueFor(exprPointer);
      }

      for (Expression parent in exprPointer.parents) {
        bool parentHasUnvisitedChildren = false;
        for (Expression child in parent.children) {
          if (!child.isVisited) {
            parentHasUnvisitedChildren = true;
          }
        }
        if (!parentHasUnvisitedChildren) {
          setValueFor(parent);

          for (Expression ancestor in parent.parents) {
            unvisited.add(ancestor);
          }
        }
      }

      unvisited.remove(exprPointer);
      if (unvisited.isNotEmpty) {
        exprPointer = unvisited.last;
      }
    } while (unvisited.isNotEmpty);

    if (varNum + 1 < model.variables.length) {
      if (check(model.variables[varNum++], varNum++)) {
        return true;
      } else {
        // SPLIT LEFT (or try false if bool)
        ValueRange splitLeft = ValueRange.splitLeft(variable.range);
        ValueRange splitRight = ValueRange.splitRight(variable.range);
        variable.range = splitLeft;
        if (check(variable, varNum)) {
          return true;
        } else {
          // SPLIT RIGHT (of fail if bool)
          variable.range = splitRight;
          return check(variable, varNum);
        }
      }
    } else {
      return false;
    }
  }

  bool setValueFor(Expression expr) {
    var tempVal;
    switch (expr.type) {
      case "Constant":
        print("${expr.type} = ${expr.value.value}");
        return true;
      case "Variable":
        if (expr.isLogic()) {
          // range for logic is always restricted to either true or false during recursion
          tempVal = expr.range.upper.value.value;
        } else {
          // num type expressions always recurse on the middle of the range
          tempVal = expr.range.mid().value.value;
        }
        break;
      case "Add":
        tempVal = expr.children[0].value.value + expr.children[1].value.value;
        break;
      case "And":
        tempVal = expr.children[0].value.value && expr.children[1].value.value;
        break;
      case "Equals":
        tempVal = expr.children[0].value.value == expr.children[1].value.value;
        break;
      case "LessThan":
        tempVal = expr.children[0].value.value < expr.children[1].value.value;
        break;
      case "GreaterThan":
        tempVal = expr.children[0].value.value > expr.children[1].value.value;
        break;
      default:
        tempVal = Values.logic(false).value;
        break;
    }
    expr.value = Values.dynamic(tempVal);
    print("${expr.type} = ${expr.value.value}");
    if(expr.range.contains(tempVal)){
      expr.isVisited = true;
      return true;
    } else {
      return false;
    }
  }
}
