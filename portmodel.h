#ifndef PORTMODEL_H
#define PORTMODEL_H

#include <QAbstractListModel>
#include "blockitem.h"

class PortModel : public QAbstractListModel
{
    Q_OBJECT

public:
    explicit PortModel(QObject *parent = nullptr);

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

    Q_INVOKABLE void addPort(int side, int position);

private:
    QVector<Port*> m_ports;
    QHash<int, QByteArray> m_roles;
    BlockItem * m_parent;
};

#endif // PORTMODEL_H
