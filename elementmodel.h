#ifndef ELEMENTMODEL_H
#define ELEMENTMODEL_H

#include <QObject>
#include <QAbstractItemModel>
#include "datasource.h"

class ElementModel : public QAbstractItemModel
{
    Q_OBJECT
public:
    Q_PROPERTY(DataSource* dataSource READ dataSource WRITE setdataSource NOTIFY dataSourceChanged)

    enum ElementRoles{
        XposRole = Qt::UserRole + 1,
        YposRole,
        ThisRole,
        TypeRole,
        RotationRole,
    };

    explicit ElementModel(QObject *parent = nullptr);

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
    void setdataSource(DataSource* newDataSource);

signals:
    void dataSourceChanged(DataSource* dataSource);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    DataSource* m_dataSource;
};

#endif // ELEMENTMODEL_H
