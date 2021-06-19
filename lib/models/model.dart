import 'expression.dart';

class Model {
  Order order = Order();

  /// single list of equation roots (highest order expression)
  List<Expression> expressions = [];

  /// points to all of the variables in the model
  List<Expression> variables = [];

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
}
