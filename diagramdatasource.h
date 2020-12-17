#ifndef DIAGRAMDATASOURCE_H
#define DIAGRAMDATASOURCE_H

#include <QObject>
#include "blockitem.h"
#include "equationsolver.h"

class DiagramDataSource : public QObject
{
    Q_OBJECT
public:
    explicit DiagramDataSource(QObject *parent = nullptr);
    ~DiagramDataSource();

    BlockItem * proxyRoot();

    void newProxyRoot(BlockItem *newProxyRoot);

    Q_INVOKABLE void appendBlock(int x = 0, int y = 0);
    Q_INVOKABLE BlockItem *thisBlock(int blockIndex);
    Q_INVOKABLE void downLevel(int modelIndex);
    Q_INVOKABLE void upLevel();
    Q_INVOKABLE void printProxyTree(BlockItem * rootItem, int depth);
    Q_INVOKABLE void printFullTree(BlockItem * rootItem, int depth);
    Q_INVOKABLE void printBlock(int blockIndex);
    Q_INVOKABLE int distanceFromRoot() const;
    Q_INVOKABLE int numChildren(int blockIndex);
    Q_INVOKABLE void deleteBlock(int blockIndex);
    Q_INVOKABLE void addPort(int blockIndex, int side, int position);
    Q_INVOKABLE int portCount(int blockIndex);
    Q_INVOKABLE int portSide(int blockIndex, int portNum);
    Q_INVOKABLE int portPosition(int blockIndex, int portNum);

    /* EXPOSING EQUATIONSOLVER FUNCTIONS AS SLOTS TO QML VIA BLOCKDATASOURCE->BLOCKMODEL */
    Q_INVOKABLE void solveEquations();

    /* FUNCTIONS AS SLOTS TO QML TO AID IN QUI OPERATIONS */
    Q_INVOKABLE int maxBlockX();
    Q_INVOKABLE int maxBlockY();

signals:
    void beginResetBlockModel();
    void endResetBlockModel();
    void beginInsertBlock();
    void endInsertBlock();
    void beginRemoveBlock(int blockIndex);
    void endRemoveBlock();
private:
    BlockItem * m_root;
    z3::context m_context;
    BlockItem * m_proxyRoot;
};

#endif // DIAGRAMDATASOURCE_H
