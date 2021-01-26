#ifndef LINKMODEL_H
#define LINKMODEL_H

#include <QAbstractItemModel>
#include "datasource.h"

class LinkModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    Q_PROPERTY(Port* proxyPort READ proxyPort WRITE setProxyPort NOTIFY proxyPortChanged)

    explicit LinkModel(QObject *parent = nullptr);
    ~LinkModel();

    enum LinkRoles{
        ThisLinkRole = Qt::UserRole + 1,
        LastPointRole,
        PortConnectedRole
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

    Port* proxyPort() const;
    void setProxyPort(Port* proxyPort);

signals:
    void proxyPortChanged(Port* proxyPort);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    Port* m_proxyPort;
};

#endif // LINKMODEL_H
