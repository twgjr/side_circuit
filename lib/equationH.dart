import 'solver.dart';

class Equation {

    String equationString;
    Expression equationExpression;

    void setEquationString( String value )
    {
        equationString = value;
        eqStrToExpr();
    }

    void eqStrToExpr()
    {
        Parser equationParser = Parser();
        equationParser.parseEquation(equationString);
        equationExpression = equationParser.expressionGraph;
    }
}
