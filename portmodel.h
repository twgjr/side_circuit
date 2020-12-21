#ifndef PORTMODEL_H
#define PORTMODEL_H

#include <QAbstractListModel>
#include "blockitem.h"
#include "proxyports.h"

class PortModel : public QAbstractListModel
{
    Q_OBJECT

public:
    Q_PROPERTY(ProxyPorts* proxyPorts READ proxyPorts WRITE setProxyPorts NOTIFY proxyPortsChanged)

    explicit PortModel(QObject *parent = nullptr);
    ~PortModel();

    enum PortRoles{
        SideRole = Qt::UserRole + 1,
        NameRole,
        PositionRole
    };

    // Basic functionality:
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    // Editable:
    bool setData(const QModelIndex &index, const QVariant &value,
                 int role = Qt::EditRole) override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    QHash<int, QByteArray> roleNames() const override;

    ProxyPorts* proxyPorts();

    void setProxyPorts(ProxyPorts* portDataSource);

signals:
    void proxyPortsChanged(ProxyPorts* portDataSource);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    ProxyPorts* m_proxyPorts;
};

#endif // PORTMODEL_H
