#ifndef PORTMODEL_H
#define PORTMODEL_H

#include <QAbstractItemModel>
#include "datasource.h"

class Block;

class PortModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    Q_PROPERTY(Block* proxyChildBlock READ proxyChildBlock WRITE setProxyChildBlock NOTIFY proxyChildBlockChanged)

    explicit PortModel(QObject *parent = nullptr);
    ~PortModel();

    enum PortStates{
        NotConnected,
        Connected,
        Error
    };

    enum PortRoles{
        ThisPort = Qt::UserRole + 1,
        SideRole,
        NameRole,
        PositionRole,
        State
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

    Block* proxyChildBlock() const;
    void setProxyChildBlock(Block* proxyChildBlock);

signals:
    void proxyChildBlockChanged(Block* proxyChildBlock);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    Block* m_proxyChildBlock;
};

#endif // PORTMODEL_H
