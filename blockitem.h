//PROVIDES STRUCTURE OF INDIVIDUAL JOKE ITEMS IN THE MODEL

#ifndef BLOCKITEM_H
#define BLOCKITEM_H

#include <QObject>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include "z3++.h"
#include "equationparser.h"
#include "equation.h"

class BlockItem : public QObject
{
    Q_OBJECT
    //Q_PROPERTY(QString category READ category WRITE setCategory NOTIFY categoryChanged)
    //Q_PROPERTY(int id READ id WRITE setId NOTIFY idChanged)
    //Q_PROPERTY(int blockXPosition READ blockXPosition WRITE setBlockXPosition NOTIFY blockXPositionChanged)
    //Q_PROPERTY(int blockYPosition READ blockYPosition WRITE setBlockYPosition NOTIFY blockYPositionChanged)
    //Q_PROPERTY(Equation equation READ equation WRITE setEquation NOTIFY equationChanged)

public:
    explicit BlockItem(z3::context * context, QObject *parent = nullptr);

    void jsonWrite(QJsonObject &json);
    void jsonRead(QJsonObject &json);

    // Category
    QString category() const;
    void setCategory(QString category);

    // Block ID
    int id() const;
    void setId(int id);

    // GUI X Position
    int blockXPosition() const;
    void setBlockXPosition(int blockXPosition);

    //GUI Y Position
    int blockYPosition() const;
    void setBlockYPosition(int blockYPosition);

    Equation * equation();
    void setEquation(QString equationString);

signals:
    void categoryChanged(QString category);
    void idChanged(int id);
    void blockXPositionChanged(int blockXPosition);
    void blockYPositionChanged(int blockYPosition);
    void equationChanged(QString equation);

private:
    QString m_category;
    int m_id;
    int m_blockXPosition;
    int m_blockYPosition;
    Equation m_equation;
    z3::context* m_solverContextReference;
};

#endif // BLOCKITEM_H
