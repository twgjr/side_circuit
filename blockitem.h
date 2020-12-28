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
//#include "portmodel.h"

class BlockItem : public QObject
{
    Q_OBJECT

    Q_PROPERTY(BlockItem* proxyRoot READ proxyRoot WRITE setProxyRoot NOTIFY proxyRootChanged)
    Q_PROPERTY(BlockItem* thisBlock READ thisBlock WRITE setThisBlock NOTIFY thisBlockChanged)

    Q_PROPERTY(QString description READ description WRITE setDescription)
    Q_PROPERTY(int id READ id)
    Q_PROPERTY(int blockXPosition READ blockXPosition WRITE setBlockXPosition)
    Q_PROPERTY(int blockYPosition READ blockYPosition WRITE setBlockYPosition)
    Q_PROPERTY(int blockWidth READ blockWidth WRITE setBlockWidth)
    Q_PROPERTY(int blockHeight READ blockHeight WRITE setblockHeight)
    Q_PROPERTY(int numChildren READ childBlockCount)
    Q_PROPERTY(QString equationString READ equationString WRITE setEquationString)

public:
    explicit BlockItem(z3::context * context, BlockItem *parentBlock,
                       QObject *parent = nullptr);
    ~BlockItem();

    enum BlockType{
        Block,
        CircuitBlock,
        EquationBlock,
        CoreBlock
    };

    // parent
    BlockItem * parentBlock();
    void setParentBlock(BlockItem *parentBlock);

    // children
    BlockItem * childBlock(int index);
    int childBlockNumber() const;
    int childBlockCount() const;
    bool appendBlockChild(BlockItem *item);
    void removeBlockChild(int modelIndex);

    // ports
    Port * portAt( int portIndex );
    void addPort(int side, int position);
    void removePort(int portIndex);
    int portCount();

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

    BlockItem* proxyRoot();
    void setProxyRoot(BlockItem* proxyRoot);

    BlockItem* thisBlock();
    void setThisBlock(BlockItem* thisBlock);

signals:
    void beginResetPortModel();
    void endResetPortModel();
    void beginInsertPort(int portIndex);
    void endInsertPort();
    void beginRemovePort(int portIndex);
    void endRemovePort();

    void proxyRootChanged(BlockItem* proxyRoot);
    void thisBlockChanged(BlockItem* thisBlock);

private:
    //object pointers
    BlockItem * m_parentItem;
    QVector<BlockItem*> m_children;
    int m_proxyChildCount;
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

    BlockItem* m_proxyRoot;
    BlockItem* m_thisBlock;
};

#endif // BLOCKITEM_H
