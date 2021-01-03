#ifndef DIAGRAMMODEL_H
#define DIAGRAMMODEL_H

#include <QObject>
#include <QAbstractItemModel>
#include <QDebug>
#include <QFile>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include "datasource.h"

class DiagramModel : public QAbstractItemModel
{
    Q_OBJECT
    Q_PROPERTY(DataSource* dataSource READ dataSource WRITE setdataSource NOTIFY dataSourceChanged)

public:

    enum BlockRoles{
        ProxyRoot = Qt::UserRole + 1,
        ThisBlock,
        DescriptionDataRole,
        IDDataRole,
        BlockXPositionRole,
        BlockYPositionRole,
        blockWidthRole,
        blockHeightRole,
        ThisRole,
        BlockRolesEnd
    };

    enum EquationRoles{
        EquationRole = Qt::UserRole + 1,
        EqXPosRole,
        EqYPosRole,
        EquationRolesEnd
    };

    explicit DiagramModel(QObject *parent = nullptr);
    ~DiagramModel();

    // QAbstractItemModel overrides
    QModelIndex index(int row,
                      int column,
                      const QModelIndex &parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex &index) const override;
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    bool setData(const QModelIndex &index,
                 const QVariant &value,
                 int role) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    QHash<int,QByteArray> roleNames() const override;

    DataSource* dataSource() const;
    void setdataSource(DataSource* blockDataSource);

signals:
    void dataSourceChanged(DataSource* newBlockDataSource);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    DataSource * m_dataSource;
};

#endif // DIAGRAMMODEL_H
