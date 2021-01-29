#ifndef PORTMODEL_H
#define PORTMODEL_H

#include <QAbstractItemModel>
#include "datasource.h"

class DiagramItem;

class PortModel : public QAbstractItemModel
{
    Q_OBJECT
    Q_PROPERTY(DiagramItem* parentItem READ parentItem WRITE setParentItem NOTIFY parentItemChanged)

public:
    enum PortRoles{
        ThisPort = Qt::UserRole + 1,
        SideRole,
        NameRole,
        PositionRole,
        AbsPointRole,
        LinkIsValidRole
    };

    explicit PortModel(QObject *parent = nullptr);
    ~PortModel();

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

    DiagramItem* parentItem() const;
    void setParentItem(DiagramItem* proxyChildItem);

signals:
    void parentItemChanged(DiagramItem* proxyChildItem);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    DiagramItem* m_parentItem;
};

#endif // PORTMODEL_H
