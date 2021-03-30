#include "resultmodel.h"

ResultModel::ResultModel(QObject *parent) : QAbstractItemModel(parent),
    m_signalConnected(false)
{
    m_roles[VarNameRole]="varName";
    m_roles[VarNumRole]="varVal";
}

QModelIndex ResultModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }

    Result *item = m_dataSource->proxyRoot()->resultAt(row);
    return createIndex(row, column, item);

    return QModelIndex();
}

QModelIndex ResultModel::parent(const QModelIndex &index) const
{
    Q_UNUSED(index);
    return QModelIndex();
}

int ResultModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }

    return m_dataSource->proxyRoot()->resultCount();
}

int ResultModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1; //columns not used
}

QVariant ResultModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()){
        return QVariant();
    }

    Result * item = m_dataSource->proxyRoot()->resultAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;

    return QVariant();
}

bool ResultModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    bool somethingChanged = false;

    Result * item = m_dataSource->proxyRoot()->resultAt(index.row());
    switch (role) {
    case VarNumRole:
        if(item->varVal() != value.toDouble()){
            item->setVarVal(value.toDouble());
        }
        break;
    case VarNameRole:
        if(item->varName() != value.toString()){
            item->setVarName(value.toString());
        }
        break;
    }

    if(somethingChanged){
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

Qt::ItemFlags ResultModel::flags(const QModelIndex &index) const
{
    if (!index.isValid()){
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}

QHash<int, QByteArray> ResultModel::roleNames() const
{
    return m_roles;
}

DataSource *ResultModel::dataSource() const
{
    return m_dataSource;
}

void ResultModel::setDataSource(DataSource *newDataSource)
{
    beginResetModel();
    if(m_dataSource && m_signalConnected){
        m_dataSource->disconnect(this);
    }

    m_dataSource = newDataSource;

    connect(m_dataSource,&DataSource::beginResetResults,this,[=](){
        beginResetModel();
    });
    connect(m_dataSource,&DataSource::endResetResults,this,[=](){
        endResetModel();
    });
    connect(m_dataSource,&DataSource::beginInsertResult,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endInsertResult,this,[=](){
        endInsertRows();
    });
    connect(m_dataSource,&DataSource::beginRemoveResult,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endRemoveResult,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}