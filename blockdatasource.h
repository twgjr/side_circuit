// CREATES A LIST OF BLOCKS AND ASSOCIATED FUNCTIONS TO MANIPULATE
// SOME FUNCTIONS NEEDED IN QML ARE CALLED FROM BLOCKMODEL TO
// ALLOW USE OF ONE QML TYPE AND CLEANER CODE FOR THE QML/C++ INTERFACE

#ifndef BLOCKDATASOURCE_H
#define BLOCKDATASOURCE_H

#include <QObject>
#include <QFile>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QDebug>
#include "blockitem.h"
#include "equationsolver.h"

class BlockDataSource : public QObject
{
    Q_OBJECT
public:
    explicit BlockDataSource(QObject *parent = nullptr);

    bool loadBlockItems(QVariant loadLocation);
    bool saveBlockItems(QVariant saveLocation);
    void appendBlockItem(BlockItem* blockItem);
    void appendBlockItem(); // made with defaults
    void appendBlockItem(const QString & blockItemCategory);
    void appendBlockItem(const QString & blockItemCategory,const int & blockItemId);
    void removeBlockItem(int index);
    void clearBlockItems();
    void removeLastBlockItem();
    QVector<BlockItem*> dataItems();

    /* FUNCTIONS USED IN BLOCKMODEL.H AS SLOTS TO QML TO AID IN QUI OPERATIONS */
    int maxBlockX();
    int maxBlockY();

    /* FUNCTIONS FOR MANIPULATING THE SOLVER */
    void solveEquations();
    EquationSolver equationSolverObj() const;

signals:
    void preItemAdded();
    void postItemAdded();
    void preItemRemoved(int index);
    void postItemRemoved();

private:
    QVector<BlockItem*> m_BlockItems;
    z3::context m_context;
};

#endif // BLOCKDATASOURCE_H
