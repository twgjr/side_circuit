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
#include "datasource.h"

class BlockModel : public QAbstractItemModel
{
    Q_OBJECT
    Q_PROPERTY(DataSource* dataSource READ dataSource WRITE setdataSource NOTIFY dataSourceChanged)

public:

    enum ModelRoles{
        DescriptionDataRole = Qt::UserRole + 1,
        IDDataRole,
        BlockXPositionRole,
        BlockYPositionRole,
        blockWidthRole,
        blockHeightRole,
        EquationRole,
        ThisRole
    };

    explicit BlockModel(QObject *parent = nullptr);
    ~BlockModel();

    // QAbstractItemModel read-only functions
    QVariant data(const QModelIndex &index, int role) const override;
    QModelIndex index(int row,
                      int column,
                      const QModelIndex &parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex &index) const override;
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;

    // QAbstractItemModel functions for editable model
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    bool setData(const QModelIndex &index,
                 const QVariant &value,
                 int role) override;

    //functions for working with roles
    QHash<int,QByteArray> roleNames() const override;
    QVariantList roles() const;
    void setRoles(QVariantList roles);

    QModelIndex qIndexOfBlock(BlockItem *item);
    BlockItem * blockFromQIndex(const QModelIndex &index) const;

    DataSource* dataSource() const;
    void setdataSource(DataSource* blockDataSource);

signals:
    void rolesChanged();
    void dataSourceChanged(DataSource* newBlockDataSource);

private:
    QHash<int, QByteArray> m_roles;
    DataSource * m_dataSource;
    bool m_signalConnected;
};

#endif // BLOCKMODEL_H
