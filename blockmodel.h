// ORGANIZES THE BLOCKS IN A FORMAT USABLE BY LISTVIEW IN QML
// SHOULD ONLY CONTAIN THINGS ASSOCIATED WITH THE QML/C++ INTERFACE
// SOME FUNCTIONS MAY BE REPEATED FROM OTHER CLASSES TO CLEANLY
// EXPOSE THEM TO QML. BETTER FOR READABILITY AND SIMPLICITY

#ifndef BLOCKMODEL_H
#define BLOCKMODEL_H

#include <QObject>
#include <QAbstractListModel>
#include <QDebug>
#include "blockdatasource.h"

class BlockModel : public QAbstractListModel
{
    Q_OBJECT
    Q_PROPERTY(BlockDataSource* blockDataSource READ blockDataSource WRITE setBlockDataSource)

    enum ModelRoles{
        CategoryDataRole = Qt::UserRole + 1,
        IDDataRole,
        BlockXPositionRole,
        BlockYPositionRole,
        EquationRole
    };

public:
    explicit BlockModel(QObject *parent = nullptr);


    /* REQUIRED FUNCTIONS FOR QABSTRACTLISTMODEL */
    // Basic functionality:
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
    // Editable:
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole);
    Qt::ItemFlags flags(const QModelIndex& index) const;
    QHash<int,QByteArray> roleNames() const;

    /* FUNCTIONS FOR SETTING BLOCKDATASOURCE */
    BlockDataSource* blockDataSource() const;
    void setBlockDataSource(BlockDataSource* blockDataSource);

    /* EXPOSING BLOCKDATASOURCE FUNCTIONS AS SLOTS TO QML VIA BLOCKMODEL */
    Q_INVOKABLE bool loadBlockItems(QVariant loadLocation){return m_blockDataSource->loadBlockItems(loadLocation);};
    Q_INVOKABLE bool saveBlockItems(QVariant saveLocation){return m_blockDataSource->saveBlockItems(saveLocation);};
    Q_INVOKABLE void appendBlockItem(){m_blockDataSource->appendBlockItem();};
    Q_INVOKABLE void clearBlockItems(){m_blockDataSource->clearBlockItems();};
    Q_INVOKABLE void removeLastBlockItem(){m_blockDataSource->removeLastBlockItem();};

    /* EXPOSING EQUATIONSOLVER FUNCTIONS AS SLOTS TO QML VIA BLOCKDATASOURCE->BLOCKMODEL */
    Q_INVOKABLE void solveEquations(){m_blockDataSource->solveEquations();};

    /* FUNCTIONS AS SLOTS TO QML TO AID IN QUI OPERATIONS */
    Q_INVOKABLE int maxBlockX(){return m_blockDataSource->maxBlockX();};
    Q_INVOKABLE int maxBlockY(){return m_blockDataSource->maxBlockY();};

private:
    /* VARIABLES FOR SETTING BLOCKDATASOURCE */
    BlockDataSource* m_blockDataSource;
    bool connectedtoBlockDataSourceSignals;
};

#endif // BLOCKMODEL_H
