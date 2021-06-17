import 'solver.dart';

class Equation {

    String equationString;
    Expression equationExpression;

    Equation();
    Equation.string(this.equationString){
        eqStrToExpr();
    }

    void eqStrToExpr() {
        Parser equationParser = Parser();
        equationParser.parseEquation(equationString);
        equationExpression = equationParser.expressionGraph;
        print("tree:");
        equationParser.printTree(equationExpression);
    }
}
