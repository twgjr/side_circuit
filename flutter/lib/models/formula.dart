import 'expression.dart';
import 'parser.dart';
import 'model.dart';

class Formula {
  Expression? _expr;
  Model model;
  List<Expression> variables = [];

  /// for adding expressions without variables
  Formula.expr(this.model,this._expr);

  Formula.string(this.model, String formulaString) {
    Parser parser = Parser(this.model,this);
    parser.parseFormula(formulaString);
    if (stringFrom(parser.formulaGraph) == formulaString) {
      this._expr = parser.formulaGraph;
    }
    for(Expression variable in this.variables){
      variable.setInitialRange();
    }
  }

  Expression get expr{
    if(_expr == null) {
      return Expression.empty(this.model);
    }
    return _expr!;
  }

  String get formulaString {
      return stringFrom(expr);
  }

  String stringFrom(Expression? expr) {
    if (expr!.children.length == 0) {
      // reached the leaves of the expression tree
      return expr.toString();
    }

    if (expr.children.length == 1) {
      if(expr.isBracket()){
        return "(" + stringFrom(expr.children[0]) + ")";
      } else {
        // is a function or unary operator sitting to left
        return expr.toString() + stringFrom(expr.children[0]) + ")";
      }
    }

    if (expr.children.length == 2) {
      return stringFrom(expr.children[0]) +
          expr.toString() +
          stringFrom(expr.children[1]);
    }

    return "";
  }
}
