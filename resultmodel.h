#ifndef RESULTMODEL_H
#define RESULTMODEL_H

#include <QObject>
#include <QAbstractItemModel>
#include "datasource.h"

class ResultModel : public QAbstractItemModel
{
    Q_OBJECT
public:
    Q_PROPERTY(DataSource* dataSource READ dataSource WRITE setDataSource NOTIFY dataSourceChanged)
    enum EquationRoles{
        ValRole = Qt::UserRole + 1,
        VarRole
    };

    explicit ResultModel(QObject *parent = nullptr);

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
    void setDataSource(DataSource* newDataSource);

signals:
    void dataSourceChanged(DataSource* dataSource);

private:
    QHash<int, QByteArray> m_roles;
    bool m_signalConnected;
    DataSource * m_dataSource;
};

#endif // RESULTMODEL_H
