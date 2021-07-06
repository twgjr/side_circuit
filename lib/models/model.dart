import 'expression.dart';
import 'values.dart';

class Model {
  Order order = Order();

  /// single list of equation roots (highest order expression)
  List<Expression> expressions = [];

  /// points to all of the variables in the model
  List<Expression> variables = [];

  /// points to all of the constants in the model
  List<Expression> constants = [];

  // the root expression that is a logical and of all expressions
  Expression? root;

  Model() {
    root = Expression.constant(this, Values.logic(true));
  }

  /// Adds variable to model if does not exist.
  /// returns pointer to variable
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
    if (expressions.isEmpty) {
      return; // keep default logic true constant for root
    }

    Expression andRoot = Expression.and(this);

    Expression iterator = andRoot;

    for (int i = 0; i < expressions.length; i++) {
      expressions[i].parents.add(iterator);
      iterator.children.add(expressions[i]);
      // even number of expressions
      if (i.isOdd) {
        Expression newAnd = Expression.and(this);
        iterator.children.add(newAnd);
        iterator = newAnd;
      }
    }

    // add a dummy true logic constant to satisfy the and expression
    // for expressions list that have odd number
    if (expressions.length.isOdd) {
      Expression constant = Expression.constant(this, Values.logic(true));
      constant.parents.add(iterator);
      iterator.children.add(constant);
    }
    for (int j = 0; j < iterator.children.length; j++) {
      print("${iterator.type} with child $j as ${iterator.children[j].type}");
    }

    root = andRoot;
  }

  void printSolution() {
    for( Expression variable in this.variables){
      print("${variable.varName} = ${variable.value.value}");
    }
  }
}
