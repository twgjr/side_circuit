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
#include "port.h"

class BlockItem : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QString description READ description WRITE setDescription)
    Q_PROPERTY(int id READ id)
    Q_PROPERTY(int blockXPosition READ blockXPosition WRITE setBlockXPosition)
    Q_PROPERTY(int blockYPosition READ blockYPosition WRITE setBlockYPosition)
    Q_PROPERTY(int blockWidth READ blockWidth WRITE setBlockWidth)
    Q_PROPERTY(int blockHeight READ blockHeight WRITE setblockHeight)
    Q_PROPERTY(int numChildren READ childCount)
    Q_PROPERTY(QString equationString READ equationString WRITE setEquationString)
    Q_PROPERTY(BlockItem * thisBlock READ thisBlock)

public:
    explicit BlockItem(z3::context * context,
                       BlockItem *parent = nullptr,
                       QObject * qobjparent = nullptr);
    ~BlockItem();

    enum BlockType{
        Block,
        CircuitBlock,
        EquationBlock,
        CoreBlock
    };

    // parent
    BlockItem * parentItem();
    void setParentItem(BlockItem *parentItem);

    // children
    BlockItem * child(int index);
    int childNumber() const;
    int childCount() const;
    bool appendChild(BlockItem *item);
    void removeChild(int modelIndex);
    int columnCount() const;

    // proxy model for diagram view
    BlockItem *proxyParent();
    void clearProxyParent();
    BlockItem * proxyChild(int index);
    int proxyChildNumber() const;
    int proxyChildCount() const;
    void clearProxyChildren();
    void appendProxyChild(BlockItem * item);
    void removeProxyChild(int modelIndex);


    // connecting blocks
    QVector<Port *> ports() const;
    void addPort(int side, int position);
    int portCount();
    int portSide(int portNum);
    int portPosition(int portNum);

    // solver
    void setContext(z3::context *context);
    z3::context *context() const;

    // data save and load
//    void jsonWrite(QJsonObject &json);
//    void jsonRead(QJsonObject &json);

    //data getters and setters
    void setBlockType(int blockType);
    int blockType() const;
    QString description() const;
    void setDescription(QString category);
    int id() const;
    int blockXPosition() const;
    void setBlockXPosition(int blockXPosition);
    int blockYPosition() const;
    void setBlockYPosition(int blockYPosition);
    Equation * equation();
    QString equationString();
    void setEquationString(QString equationString);
    int blockWidth() const;
    int blockHeight() const;
    void setBlockWidth(int blockWidth);
    void setblockHeight(int blockHeight);


    BlockItem* thisBlock() const
    {
        return m_thisBlock;
    }

signals:
private:
    //object pointers
    BlockItem * m_parentItem;
    QVector<BlockItem*> m_children;
    BlockItem * m_proxyParent;
    QVector<BlockItem*> m_proxyChildren;
    QVector<Port*> m_ports;

    z3::context* m_context;

    //Data
    int m_blockType;
    QString m_description;
    int m_blockXPosition;
    int m_blockYPosition;
    Equation m_equation;
    int m_blockWidth;
    int m_blockHeight;

    BlockItem* m_thisBlock;
};

#endif // BLOCKITEM_H
