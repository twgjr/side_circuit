import 'expression.dart';
import 'values.dart';
import 'formula.dart';

class Model {
  Order order = Order();

  /// single list of equation roots (highest order expression)
  List<Formula> formulas = [];

  /// points to all of the variables in the model
  List<Expression> variables = [];

  /// points to all of the constants in the model
  List<Expression> constants = [];

  // the root expression that is a logical and of all expressions
  Expression? root;

  Model() {
    root = Expression.constant(this, Value.logic(true));
  }

  void addFormula(Expression expr) {
    formulas.add(Formula.expr(this, expr));
  }

  /// Adds variable to model if does not exist.
  /// returns pointer to variable
  //@todo choose the variable isBool based on it's parent
  Expression addVariable(String variableID) {
    for (Expression variable in this.variables) {
      if (variable.varName == variableID) {
        return variable;
      }
    }
    Expression varToAdd = Expression.variable(this, variableID);
    this.variables.add(varToAdd);
    return varToAdd;
  }

  void buildRoot() {
    print("building solver tree");
    if (formulas.isEmpty) {
      return; // keep default logic true constant for root
    }

    Expression andRoot = Expression.solverAnd(this);

    Expression iterator = andRoot;

    for (Formula formula in this.formulas) {
      print(formula.expr.toString());
    }

    if (this.formulas.length > 0) {
      print("adding 1st formula");
      this.formulas[0].expr.parents.add(iterator);
      iterator.children.add(this.formulas[0].expr);
    }

    if (this.formulas.length == 2) {
      print("adding 2nd formula");
      this.formulas[1].expr.parents.add(iterator);
      iterator.children.add(this.formulas[1].expr);
    } else {
      for (int i = 1; i < this.formulas.length; i++) {
        print("adding ${i}th formula");
        Expression newAnd = Expression.solverAnd(this);
        newAnd.parents.add(iterator);
        iterator.children.add(newAnd);
        formulas[i].expr.parents.add(newAnd);
        newAnd.children.add(formulas[i].expr);
        iterator = newAnd;
      }

      for (int j = 0; j < iterator.children.length; j++) {
        print("2, ${iterator.type}: child $j = ${iterator.children[j].type}");
        if (iterator.parents.isNotEmpty) {
          print("    parent ${iterator.parents[0].toString()}");
        }
      }

      // add a dummy true logic constant to satisfy the and expression
      // for formula list that has odd number
      if (this.formulas.length.isOdd) {
        print("adding a dummy true");
        Expression constant = Expression.constant(this, Value.logic(true));
        constant.parents.add(iterator);
        iterator.children.add(constant);
      }
    }

    for (int j = 0; j < iterator.children.length; j++) {
      print("3, ${iterator.type}: child $j = ${iterator.children[j].type}");
      if (iterator.parents.isNotEmpty) {
        print("    parent ${iterator.parents[0].toString()}");
      }
    }

    root = andRoot;
  }

  void printSolution() {
    for (Expression variable in this.variables) {
      print("${variable.varName} = ${variable.value.stored}");
    }
  }
}