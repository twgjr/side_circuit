//PROVIDES STRUCTURE OF INDIVIDUAL BLOCK ITEMS IN THE MODEL

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
#include "link.h"
#include "port.h"

class BlockItem : public QObject
{
    Q_OBJECT

    //***TODO*** expand equation into list of equations using QVector<QVariant> dataList
    Q_PROPERTY(QString category READ category WRITE setCategory)
    Q_PROPERTY(int id READ id)
    Q_PROPERTY(int blockXPosition READ blockXPosition WRITE setBlockXPosition)
    Q_PROPERTY(int blockYPosition READ blockYPosition WRITE setBlockYPosition)
    Q_PROPERTY(int numChildren READ childCount)
    Q_PROPERTY(QString equationString READ equationString WRITE setEquationString)

public:
    explicit BlockItem(z3::context * context,
                       BlockItem *parent = nullptr,
                       QObject * qobjparent = nullptr);
    ~BlockItem();

    enum ModelRoles{
        CategoryDataRole = Qt::UserRole + 1,
        IDDataRole,
        BlockXPositionRole,
        BlockYPositionRole,
        EquationRole,
    };

    enum BlockType{
        MainBlock,
        EquationBlock,
        CoreBlock
    };

    //parent
    BlockItem * parentItem();

    //children
    BlockItem * child(int index);
    int childNumber() const;
    int childCount() const;
    bool insertChildren(int position, int count, int columns = 0);
    bool removeChildren(int position, int count);
    bool appendChild(BlockItem *item);

    //// Generic dataList columns (do not need column data structure)
    //// Good framework for maintaining lists inside this block (e.g. list
    //// of QVector<Equation*>, variables, ports, etc.
    int columnCount() const;

    //data getters and setters
    void jsonWrite(QJsonObject &json);
    void jsonRead(QJsonObject &json);

    QString category() const;
    void setCategory(QString category);

    int id() const;

    int blockXPosition() const;
    void setBlockXPosition(int blockXPosition);

    int blockYPosition() const;
    void setBlockYPosition(int blockYPosition);

    Equation * equation();

    QString equationString();
    void setEquationString(QString equationString);

    BlockItem *realModelPointer();
    void setRealModelPointer(BlockItem *realModelPointer);

signals:
private:
    QString m_category;
    int m_blockXPosition;
    int m_blockYPosition;
    Equation m_equation;
    z3::context* m_context;

    QVector<BlockItem*> m_children;
    BlockItem * m_parent;

    QVector<Port*> m_ports;
    QVector<Link*> m_links;

    BlockItem * m_realModelPointer; //stores actual pointer to this item's complete model if proxy used
};

#endif // BLOCKITEM_H
