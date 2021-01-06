#ifndef DIAGRAMMODEL_H
#define DIAGRAMMODEL_H

#include <QObject>
#include <QAbstractItemModel>
#include "datasource.h"

class DiagramModel : public QAbstractItemModel
{
    Q_OBJECT
    Q_PROPERTY(DataSource* dataSource READ dataSource WRITE setdataSource NOTIFY dataSourceChanged)

public:

    enum DiagramRoles{
        ProxyRoot = Qt::UserRole + 1,
        ThisRole,
        TypeRole,
        DescRole,
        IdRole,
        XposRole,
        YposRole,
        WidthRole,
        HeightRole,
        EquationRole
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