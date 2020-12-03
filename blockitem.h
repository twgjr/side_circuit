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

    Q_PROPERTY(QString category READ category WRITE setCategory NOTIFY categoryChanged)
    Q_PROPERTY(int id READ id WRITE setId NOTIFY idChanged)
    Q_PROPERTY(int blockXPosition READ blockXPosition WRITE setBlockXPosition NOTIFY blockXPositionChanged)
    Q_PROPERTY(int blockYPosition READ blockYPosition WRITE setBlockYPosition NOTIFY blockYPositionChanged)
    Q_PROPERTY(QString equationString READ equationString WRITE setEquationString NOTIFY equationStringChanged)

public:
    explicit BlockItem(z3::context * context, QObject *parent = nullptr);

    Q_INVOKABLE BlockItem * parentItem() const;
    bool insertItem(BlockItem *item, int pos = -1);
    BlockItem * child(int index) const;
    void clear();
    Q_INVOKABLE int pos() const;
    Q_INVOKABLE int count() const;

    void jsonWrite(QJsonObject &json);
    void jsonRead(QJsonObject &json);

    QString category() const;
    void setCategory(QString category);
    int id() const;
    void setId(int id);
    int blockXPosition() const;
    void setBlockXPosition(int blockXPosition);
    int blockYPosition() const;
    void setBlockYPosition(int blockYPosition);
    Equation * equation();
    void setEquation(QString equationString);

    QString equationString();
    void setEquationString(QString equationString);

signals:
    void categoryChanged(QString category);
    void idChanged(int id);
    void blockXPositionChanged(int blockXPosition);
    void blockYPositionChanged(int blockYPosition);
    void equationChanged(QString equation);
    void equationStringChanged(QString equationString);

private:
    QString m_category;
    int m_id;
    int m_blockXPosition;
    int m_blockYPosition;
    Equation m_equation;
    z3::context* m_solverContextReference;

    QVector<BlockItem*> m_children;
    BlockItem * m_parent;
};

#endif // BLOCKITEM_H
