import 'expression.dart';
import 'parser.dart';
import 'model.dart';

class Equation {

    String equationString;
    Expression equationExpression;
    Model model;

    Equation(this.model);
    Equation.string(this.model,this.equationString){
        eqStrToExpr();
    }

    void eqStrToExpr() {
        Parser equationParser = Parser(this.model);
        equationParser.parseEquation(equationString);
        equationExpression = equationParser.expressionGraph;
        //print("parsed:");
        equationParser.printTree(equationExpression,null);
    }
}
