import 'solver.dart';

class Equation {

    String equationString;
    Expression equationExpression;

    Equation();
    Equation.string(this.equationString);

    void setEquationString( String value ) {
        equationString = value;
        eqStrToExpr();
    }

    void eqStrToExpr() {
        Parser equationParser = Parser();
        equationParser.parseEquation(equationString);
        equationExpression = equationParser.expressionGraph;
    }
}
