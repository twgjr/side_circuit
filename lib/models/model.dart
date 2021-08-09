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

  void addFormula(Expression expr){
    formulas.add(Formula.expr(this,expr));
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
    if (formulas.isEmpty) {
      return; // keep default logic true constant for root
    }

    Expression andRoot = Expression.solverAnd(this);

    Expression iterator = andRoot;

    for (int i = 0; i < formulas.length; i++) {
      formulas[i].expr.parents.add(iterator);
      iterator.children.add(formulas[i].expr);
      // even number of expressions
      if (i.isOdd) {
        Expression newAnd = Expression.solverAnd(this);
        iterator.children.add(newAnd);
        iterator = newAnd;
      }
    }

    // add a dummy true logic constant to satisfy the and expression
    // for expressions list that have odd number
    if (formulas.length.isOdd) {
      Expression constant = Expression.constant(this, Value.logic(true));
      constant.parents.add(iterator);
      iterator.children.add(constant);
    }
    for (int j = 0; j < iterator.children.length; j++) {
      print("${iterator.type} child $j = ${iterator.children[j].type}");
    }
    root = andRoot;
  }

  void printSolution() {
    for( Expression variable in this.variables){
      print("${variable.varName} = ${variable.value.stored}");
    }
  }
}
