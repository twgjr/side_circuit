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
//#include "portmodel.h"

class DataSource;

class Block : public QObject
{
    Q_OBJECT

    Q_PROPERTY(Block* proxyRoot READ proxyRoot WRITE setProxyRoot NOTIFY proxyRootChanged)
    Q_PROPERTY(Block* thisBlock READ thisBlock WRITE setThisBlock NOTIFY thisBlockChanged)

    Q_PROPERTY(QString description READ description WRITE setDescription)
    Q_PROPERTY(int id READ id)
    Q_PROPERTY(int blockXPosition READ blockXPosition WRITE setBlockXPosition)
    Q_PROPERTY(int blockYPosition READ blockYPosition WRITE setBlockYPosition)
    Q_PROPERTY(int blockWidth READ blockWidth WRITE setBlockWidth)
    Q_PROPERTY(int blockHeight READ blockHeight WRITE setblockHeight)
    Q_PROPERTY(int numChildren READ childBlockCount)
    Q_PROPERTY(QString equationString READ equationString WRITE setEquationString)

public:
    explicit Block(z3::context * context, Block *parentBlock,
                       QObject *parent = nullptr);
    ~Block();

    // parent
    Block * parentBlock();
    void setParentBlock(Block *parentBlock);

    int diagramItemCount();

    // block children
    Block * childBlockAt(int index);
    int childBlockNumber() const;
    int childBlockCount() const;
    void addBlockChild(int x, int y);
    void removeBlockChild(int modelIndex);

    // ports
    Port * portAt( int portIndex );
    void addPort(int side, int position);
    void removePort(int portIndex);
    int portCount();

    // Equation children
    QVector<Equation*> equations();
    int equationCount();
    Equation* childEquationAt(int index);
    void addEquation(int x, int y);
    void removeEquation(int index);
    //Equation * equation();
    //QString equationString();
    //void setEquationString(QString equationString);

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
    int blockWidth() const;
    int blockHeight() const;
    void setBlockWidth(int blockWidth);
    void setblockHeight(int blockHeight);

    Block* proxyRoot();
    void setProxyRoot(Block* proxyRoot);

    Block* thisBlock();
    void setThisBlock(Block* thisBlock);

signals:
    void beginResetBlock();
    void endResetBlock();
    void beginInsertPort(int portIndex);
    void endInsertPort();
    void beginRemovePort(int portIndex);
    void endRemovePort();

    void proxyRootChanged(Block* proxyRoot);
    void thisBlockChanged(Block* thisBlock);

private:
    //object pointers
    Block * m_parentItem;
    QVector<Block*> m_blockChildren;
    int m_proxyChildCount;
    QVector<Port*> m_ports;
    QVector<Equation*> m_equationChildren;

    z3::context* m_context;

    //Data
    QString m_description;
    int m_blockXPosition;
    int m_blockYPosition;
    //Equation m_equation;
    int m_blockWidth;
    int m_blockHeight;

    Block* m_proxyRoot;
    Block* m_thisBlock;
};

#endif // BLOCK_H
