#ifndef LINKMODEL_H
#define LINKMODEL_H

#include <QAbstractListModel>
#include "dsport.h"

class LinkModel : public QAbstractListModel
{
    Q_OBJECT

public:
    Q_PROPERTY(DSPort* dsPort READ dsPort WRITE setdsPort NOTIFY dsPortChanged)

    explicit LinkModel(QObject *parent = nullptr);
    ~LinkModel();

    enum LinkRoles{
        StartXRole = Qt::UserRole + 1,
        StartYRole,
        EndXRole,
        EndYRole
    };

    // Basic functionality:
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    // Editable:
    bool setData(const QModelIndex &index, const QVariant &value,
                 int role = Qt::EditRole) override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    QHash<int, QByteArray> roleNames() const override;

    DSPort* dsPort();
    void setdsPort(DSPort* proxyPort);

signals:
    void dsPortChanged(DSPort* proxyPort);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    DSPort* m_dsPort;

};

#endif // LINKMODEL_H
