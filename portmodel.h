#ifndef PORTMODEL_H
#define PORTMODEL_H

#include <QAbstractItemModel>
#include "datasource.h"

class PortModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    Q_PROPERTY(BlockItem* proxyChildBlock READ proxyChildBlock WRITE setProxyChildBlock NOTIFY proxyChildBlockChanged)   

    explicit PortModel(QObject *parent = nullptr);
    ~PortModel();

    enum PortRoles{
        ThisPort = Qt::UserRole + 1,
        SideRole,
        NameRole,
        PositionRole
    };

    // QAbstractItemModel overrides
    QModelIndex index(int row,
                      int column,
                      const QModelIndex &parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex &index) const override;
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex &index, const QVariant &value,
                 int role = Qt::EditRole) override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    QHash<int, QByteArray> roleNames() const override;

    BlockItem* proxyChildBlock() const;
    void setProxyChildBlock(BlockItem* proxyChildBlock);

signals:
    void proxyChildBlockChanged(BlockItem* proxyChildBlock);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    BlockItem* m_proxyChildBlock;
};

#endif // PORTMODEL_H
