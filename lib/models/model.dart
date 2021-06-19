import 'expression.dart';

class Model {
  Order order = Order();
  List<Expression> expressions; // single list of Expression roots for the Model
  List<Expression> variables; //points to all of the variables in the model
}