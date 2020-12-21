#ifndef DATASOURCE_H
#define DATASOURCE_H

#include <QObject>
#include "blockitem.h"
#include "equationsolver.h"

class DataSource : public QObject
{
    Q_OBJECT
public:
    explicit DataSource(QObject *parent = nullptr);
    ~DataSource();

    BlockItem * proxyRoot();
    Q_INVOKABLE BlockItem * proxyChild(int blockIndex);

    void newProxyRoot(BlockItem *newProxyRoot);

    Q_INVOKABLE void appendBlock(int x = 0, int y = 0);
    Q_INVOKABLE void downLevel(int modelIndex);
    Q_INVOKABLE void upLevel();
    Q_INVOKABLE void printProxyTree(BlockItem * rootItem, int depth);
    Q_INVOKABLE void printFullTree(BlockItem * rootItem, int depth);
    Q_INVOKABLE void printBlock(int blockIndex);
    Q_INVOKABLE int distanceFromRoot() const;
    Q_INVOKABLE int numChildren(int blockIndex);
    Q_INVOKABLE void deleteBlock(int blockIndex);

    Q_INVOKABLE void addPort(int blockIndex, int side, int position);

    /* EXPOSING EQUATIONSOLVER FUNCTIONS AS SLOTS TO QML VIA BLOCKDATASOURCE->BLOCKMODEL */
    Q_INVOKABLE void solveEquations();

    /* FUNCTIONS AS SLOTS TO QML TO AID IN QUI OPERATIONS */
    Q_INVOKABLE int maxBlockX();
    Q_INVOKABLE int maxBlockY();

signals:
    void beginResetBlockModel();
    void endResetBlockModel();
    void beginInsertBlock(int blockIndex);
    void endInsertBlock();
    void beginRemoveBlock(int blockIndex);
    void endRemoveBlock();

private:
    BlockItem * m_root;
    z3::context m_context;
    BlockItem * m_proxyRoot;
};

#endif // DATASOURCE_H
