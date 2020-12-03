// ORGANIZES THE BLOCKS IN A FORMAT USABLE BY LISTVIEW IN QML
// SHOULD ONLY CONTAIN THINGS ASSOCIATED WITH THE QML/C++ INTERFACE
// SOME FUNCTIONS MAY BE REPEATED FROM OTHER CLASSES TO CLEANLY
// EXPOSE THEM TO QML. BETTER FOR READABILITY AND SIMPLICITY

#ifndef BLOCKMODEL_H
#define BLOCKMODEL_H

#include <QObject>
#include <QAbstractItemModel>
#include <QDebug>
#include <QFile>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include "blockitem.h"
#include "equationsolver.h"

class BlockModel : public QAbstractItemModel
{
    Q_OBJECT
    //Q_PROPERTY(BlockDataSource* blockDataSource READ blockDataSource WRITE setBlockDataSource)
    Q_PROPERTY(QVariantList roles READ roles WRITE setRoles NOTIFY rolesChanged)

    enum ModelRoles{
        CategoryDataRole = Qt::UserRole + 1,
        IDDataRole,
        BlockXPositionRole,
        BlockYPositionRole,
        EquationRole
    };

public:
    explicit BlockModel(QObject *parent = nullptr);
    ~BlockModel() override;

    /* REQUIRED FUNCTIONS FOR QABSTRACTITEMMODEL */
    // Basic functionality:
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    // Editable:
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex &index) const override;
    QHash<int,QByteArray> roleNames() const override;

    QVariantList roles() const;
    void setRoles(QVariantList roles);

    Q_INVOKABLE QModelIndex indexFromElement(BlockItem *item);
    Q_INVOKABLE bool insertElement(BlockItem *item, const QModelIndex &parent = QModelIndex(), int pos = -1);

    BlockItem *elementFromIndex(const QModelIndex &index) const;

    /* FUNCTIONS FOR SETTING BLOCKDATASOURCE */
    //BlockDataSource* blockDataSource() const;
    //void setBlockDataSource(BlockDataSource* blockDataSource);

    /* EXPOSING BLOCKDATASOURCE FUNCTIONS AS SLOTS TO QML VIA BLOCKMODEL */
    //Q_INVOKABLE bool loadBlockItems(QVariant loadLocation);
   // Q_INVOKABLE bool saveBlockItems(QVariant saveLocation);
    Q_INVOKABLE void appendBlockItem();
    //Q_INVOKABLE void clearBlockItems();
    //Q_INVOKABLE void removeLastBlockItem();

    /* EXPOSING EQUATIONSOLVER FUNCTIONS AS SLOTS TO QML VIA BLOCKDATASOURCE->BLOCKMODEL */
    //Q_INVOKABLE void solveEquations();

    /* FUNCTIONS AS SLOTS TO QML TO AID IN QUI OPERATIONS */
    //Q_INVOKABLE int maxBlockX();
    //Q_INVOKABLE int maxBlockY();

signals:
    void rolesChanged();

private:
    /* VARIABLES FOR SETTING BLOCKDATASOURCE */
    //BlockDataSource* m_blockDataSource;
    //bool connectedtoBlockDataSourceSignals;
    BlockItem * m_root;
    //QVariantList m_roles;
    QHash<int, QByteArray> m_roles;
    z3::context m_context;
};

#endif // BLOCKMODEL_H
