//#include "z3++.h"
import 'parser.dart';

class Equation : public QObject
{
    Q_OBJECT
public:
    Q_PROPERTY(QString equationString READ equationString WRITE setEquationString NOTIFY equationStringChanged)

    explicit Equation(z3::context * context, QObject *parent = nullptr);
    ~Equation();

    //special functions
    void printExprInfo();
    z3::expr getEquationExpression();
    void setEquationExpression(z3::expr equationExpression);
    void eqStrToExpr();

    // setters with qProperty
    void setEquationString(QString value);

    //getters with qProperty
    QString equationString() const;

signals:
    void equationStringChanged(QString equationString);

private:
    z3::context * m_equationContext;
    QString m_equationString;
    z3::expr m_equationExpression;
};

#endif // EQUATION_H
