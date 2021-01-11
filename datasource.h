#ifndef DATASOURCE_H
#define DATASOURCE_H

#include <QObject>
#include <QDebug>
#include <QFile>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include "block.h"
#include "equationsolver.h"

class Block;

class DataSource : public QObject
{
    Q_OBJECT
public:
    explicit DataSource(QObject *parent = nullptr);
    ~DataSource();

    Block * proxyRoot();
    Q_INVOKABLE Block * proxyChild(int blockIndex);
    Q_INVOKABLE Port * proxyPort(int blockIndex, int portIndex);

    void newProxyRoot(Block *newProxyRoot);

    Q_INVOKABLE void appendBlock(int x = 0, int y = 0);
    Q_INVOKABLE void deleteBlock(int blockIndex);
    Q_INVOKABLE void addEquation();
    Q_INVOKABLE void deleteEquation(int index);
    Q_INVOKABLE void downLevel(int modelIndex);
    Q_INVOKABLE void upLevel();
    Q_INVOKABLE void printFullTree(Block * rootItem, int depth);
    Q_INVOKABLE void printBlock(int blockIndex);
    Q_INVOKABLE int distanceFromRoot() const;

    Q_INVOKABLE void addPort( int blockIndex, int side, int position );
    Q_INVOKABLE void deletePort( int blockIndex, int portIndex );

    Q_INVOKABLE void startLink( int blockIndex, int portIndex );
    Q_INVOKABLE void deleteLink( int blockIndex, int portIndex, int linkIndex );
    Q_INVOKABLE void endLink(int blockIndex, int portIndex, Link* endLink);

    /* EXPOSING EQUATIONSOLVER FUNCTIONS AS SLOTS TO QML VIA BLOCKDATASOURCE->BLOCKMODEL */
    Q_INVOKABLE void solveEquations();

    /* FUNCTIONS AS SLOTS TO QML TO AID IN QUI OPERATIONS */
    Q_INVOKABLE int maxBlockX();
    Q_INVOKABLE int maxBlockY();

signals:
    //blocks
    void beginResetDiagram();
    void endResetDiagram();
    void beginInsertDiagramItem(int blockIndex);
    void endInsertDiagramItem();
    void beginRemoveDiagramItem(int blockIndex);
    void endRemoveDiagramItem();

    void beginResetEquations();
    void endResetEquations();
    void beginInsertEquation(int index);
    void endInsertEquation();
    void beginRemoveEquation(int index);
    void endRemoveEquation();

    void beginResetResults();
    void endResetResults();
    void beginInsertResult(int index);
    void endInsertResult();
    void beginRemoveResult(int index);
    void endRemoveResult();

private:
    Block * m_root;
    z3::context m_context;
    Block * m_proxyRoot;
};

#endif // DATASOURCE_H
