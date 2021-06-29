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
  Expression root;

  Model() {
    root = Expression.constant(this, Values.logic(true));
  }

  /// Adds variable to model if does not exist.
  /// returns pointer to variable
  Expression addVariable(String variableID) {
    bool variableExists = false;
    Expression tempVar;
    for (Expression variable in this.variables) {
      if (variable.varName == variableID) {
        variableExists = true;
        tempVar = variable;
        break;
      }
    }
    if (!variableExists) {
      tempVar = Expression(this);
      tempVar.type = "Variable";
      tempVar.varName = variableID;
      this.variables.add(tempVar);
    }
    return tempVar;
  }

  void buildRoot() {
    if (expressions.isEmpty) {
      return; // keep default logic true constant for root
    }

    Expression andRoot = Expression.and(this);

    Expression iterator = andRoot;

    for (int i = 0; i < expressions.length; i++) {
      // odd number of expressions
      if (i.isEven) {
        iterator.children.add(expressions[i]);
      }
      // even number of expressions
      if (i.isOdd) {
        Expression newAnd = Expression.and(this);
        iterator.children.add(newAnd);
        iterator = newAnd;
        iterator.children.add(expressions[i]);
      }

    }

    // add a dummy true logic constant to satisfy the and expression
    // for expressions list that have odd number
    if (expressions.length.isOdd) {
      iterator.children.add(Expression.constant(this, Values.logic(true)));
    }
    for(int j =0; j<iterator.children.length;j++) {
      print("$j is ${iterator.type} with child ${iterator.children[j].type}");
    }

    root = andRoot;
  }
}
