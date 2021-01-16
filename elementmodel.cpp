#include "elementmodel.h"

ElementModel::ElementModel(QObject *parent) : QAbstractItemModel(parent)
{
    m_roles[XposRole]="xPos";
    m_roles[YposRole]="yPos";
    m_roles[ThisRole]="thisItem";
    m_roles[TypeRole]="type";
    m_roles[RotationRole]="rotation";
}

QModelIndex ElementModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }

    Element *childElementItem = m_dataSource->proxyRoot()->elementAt(row);
    return createIndex(row, column, childElementItem);

    return QModelIndex();
}

QModelIndex ElementModel::parent(const QModelIndex &index) const
{
    Q_UNUSED(index);
    return QModelIndex();
}

int ElementModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }

    int elementCount = m_dataSource->proxyRoot()->elementCount();

    return elementCount;
}

int ElementModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1; //columns not used
}

QVariant ElementModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()){
        return QVariant();
    }

    Element * item = m_dataSource->proxyRoot()->elementAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;
}

bool ElementModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    bool somethingChanged = false;

    Element * elementItem = m_dataSource->proxyRoot()->elementAt(index.row());
    switch (role) {
    case XposRole:
        if(elementItem->xPos() != value.toInt()){
            elementItem->setXPos(value.toInt());
        }
        break;
    case YposRole:
        if(elementItem->yPos() != value.toInt()){
            elementItem->setYPos(value.toInt());
        }
        break;
    case TypeRole:
        if(elementItem->type() != value.toInt()){
            elementItem->setType(value.toInt());
        }
        break;
    case RotationRole:
        if(elementItem->rotation() != value.toInt()){
            elementItem->setRotation(value.toInt());
        }
        break;
    }

    if(somethingChanged){
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

Qt::ItemFlags ElementModel::flags(const QModelIndex &index) const
{
    if (!index.isValid()){
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}

QHash<int, QByteArray> ElementModel::roleNames() const
{
    return m_roles;
}

DataSource *ElementModel::dataSource() const
{
    return m_dataSource;
}

void ElementModel::setdataSource(DataSource *newDataSource)
{
    beginResetModel();
    //    if(m_dataSource && m_signalConnected){
    //        m_dataSource->disconnect(this);
    //    }

    m_dataSource = newDataSource;

    connect(m_dataSource,&DataSource::beginResetDiagram,this,[=](){
        beginResetModel();
    });
    connect(m_dataSource,&DataSource::endResetDiagram,this,[=](){
        endResetModel();
    });
    connect(m_dataSource,&DataSource::beginInsertElement,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endInsertElement,this,[=](){
        endInsertRows();
    });
    connect(m_dataSource,&DataSource::beginRemoveElement,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endRemoveElement,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
