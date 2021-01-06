#ifndef EQUATION_H
#define EQUATION_H

#include <QObject>
#include "z3++.h"
#include <QDebug>
#include "equationparser.h"

class Equation : public QObject
{
    Q_OBJECT
public:
    Q_PROPERTY(int xPos READ xPos WRITE setXPos NOTIFY xPosChanged)
    Q_PROPERTY(int yPos READ yPos WRITE setYPos NOTIFY yPosChanged)
    Q_PROPERTY(int itemWidth READ itemWidth WRITE setItemWidth NOTIFY itemWidthChanged)
    Q_PROPERTY(int itemHeight READ itemHeight WRITE setItemHeight NOTIFY itemHeightChanged)
    Q_PROPERTY(QString type READ type WRITE setType NOTIFY typeChanged)
    Q_PROPERTY(QString equationString READ equationString WRITE setEquationString NOTIFY equationStringChanged)

    explicit Equation(z3::context * context, QObject *parent = nullptr);
    ~Equation();

    //special functions
    void printExprInfo();
    z3::expr getEquationExpression();
    void setEquationExpression(z3::expr equationExpression);
    void eqStrToExpr();

    // setters with qProperty
    void setXPos(int eqXPos);
    void setYPos(int eqYPos);
    void setItemWidth(int itemWidth);
    void setItemHeight(int itemHeight);
    void setType(QString type);
    void setEquationString(QString value);

    //getters with qProperty
    int xPos() const;
    int yPos() const;
    int itemWidth() const;
    int itemHeight() const;
    QString type() const;
    QString equationString() const;

signals:
    void xPosChanged(int eqXPos);
    void yPosChanged(int eqYPos);
    void typeChanged(QString type);
    void equationStringChanged(QString equationString);
    void itemWidthChanged(int itemWidth);
    void itemHeightChanged(int itemHeight);

private:
    z3::context * m_equationContext;
    QString m_type;
    QString m_equationString;
    z3::expr m_equationExpression;
    int m_xPos;
    int m_yPos;
    int m_itemWidth;
    int m_itemHeight;
};

#endif // EQUATION_H
