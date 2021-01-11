//PROVIDES STRUCTURE OF INDIVIDUAL BLOCK ITEMS IN THE MODEL

#ifndef BLOCK_H
#define BLOCK_H

#include <QObject>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include "z3++.h"
#include "equationparser.h"
#include "equation.h"
#include "port.h"
#include "result.h"

class DataSource;
class Result;
class Block : public QObject
{
    Q_OBJECT

    Q_PROPERTY(Block* proxyRoot READ proxyRoot WRITE setProxyRoot NOTIFY proxyRootChanged)
    Q_PROPERTY(Block* thisItem READ thisItem WRITE setThisItem NOTIFY thisItemChanged)
    Q_PROPERTY(QString type READ type WRITE setType NOTIFY typeChanged)
    Q_PROPERTY(QString description READ description WRITE setDescription)
    Q_PROPERTY(int id READ id)
    Q_PROPERTY(int xPos READ xPos WRITE setXPos)
    Q_PROPERTY(int yPos READ yPos WRITE setYPos)
    Q_PROPERTY(int itemWidth READ itemWidth WRITE setItemWidth)
    Q_PROPERTY(int itemHeight READ itemHeight WRITE setblockHeight)
    Q_PROPERTY(int numChildren READ childBlockCount)


public:
    explicit Block(z3::context * context, Block *parentBlock,
                   QObject *parent = nullptr);
    ~Block();

    // parent
    Block * parentBlock();
    void setParentBlock(Block *parentBlock);

    // Block children
    Block * childBlockAt(int index);
    int childBlockNumber() const;
    int childBlockCount() const;
    void addBlockChild(int x, int y);
    void removeBlockChild(int modelIndex);

    // Ports
    Port * portAt( int portIndex );
    void addPort(int side, int position);
    void removePort(int portIndex);
    int portCount();

    // Equation children
    //QVector<Equation*> equations();
    int equationCount();
    Equation* equationAt(int index);
    void addEquation();
    void removeEquation(int index);

    // z3 solver and results
    void setContext(z3::context *context);
    z3::context *context() const;
    int resultCount();
    Result* resultAt(int index);
    void addResult(QString variable, double result);
    void clearResults();

    // data save and load
    //    void jsonWrite(QJsonObject &json);
    //    void jsonRead(QJsonObject &json);

    //getters with qProperty
    int blockType() const;
    QString description() const;
    int id() const;
    int xPos() const;
    int yPos() const;
    int itemWidth() const;
    int itemHeight() const;
    Block* proxyRoot();
    Block* thisItem();
    QString type() const;

    //setters with qProperty
    void setBlockType(int blockType);
    void setDescription(QString category);
    void setXPos(int blockXPosition);
    void setYPos(int blockYPosition);
    void setItemWidth(int blockWidth);
    void setblockHeight(int blockHeight);
    void setProxyRoot(Block* proxyRoot);
    void setThisItem(Block* thisBlock);
    void setType(QString type);

signals:
    // qAbstractItemModel signals
    void beginResetBlock();
    void endResetBlock();
    void beginInsertPort(int portIndex);
    void endInsertPort();
    void beginRemovePort(int portIndex);
    void endRemovePort();

    // qProperty signals
    void proxyRootChanged(Block* proxyRoot);
    void thisItemChanged(Block* thisBlock);
    void typeChanged(QString type);

private:
    //object pointers
    Block * m_parentItem;
    QVector<Block*> m_blockChildren;
    QVector<Port*> m_ports;
    QVector<Equation*> m_equationChildren;
    QVector<Result*> m_results;

    z3::context* m_context;

    //Data
    Block* m_proxyRoot;
    Block* m_thisItem;
    QString m_type;
    QString m_description;
    int m_xPos;
    int m_yPos;
    int m_itemWidth;
    int m_itemHeight;
};

#endif // BLOCK_H
