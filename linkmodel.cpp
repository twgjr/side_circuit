#include "linkmodel.h"

LinkModel::LinkModel(QObject *parent) : QAbstractItemModel(parent),
    m_signalConnected(false)
{
    //set the basic roles for access of item properties in QML
    m_roles[StartXRole]="startX";
    m_roles[StartYRole]="startY";
    m_roles[EndXRole]="endX";
    m_roles[EndYRole]="endY";

    qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

LinkModel::~LinkModel()
{
    qDebug()<<"Deleted: "<<this;
}

QModelIndex LinkModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }
//    BlockItem *proxyParentItem = blockFromQIndex(parent);
//    BlockItem *proxyChildItem = proxyParentItem->child(row);
    Link * linkItem = m_proxyPort->linkAt(row);
    if (linkItem){
        return createIndex(row, column, linkItem);
    }

    return QModelIndex();
}

QModelIndex LinkModel::parent(const QModelIndex &index) const
{
    return QModelIndex();
//    if (!index.isValid()){
//        // the root index
//        return QModelIndex();
//    }
//    BlockItem *proxyChildItem = static_cast<BlockItem*>(index.internalPointer());
//    BlockItem *proxyParentItem = static_cast<BlockItem *>(proxyChildItem->proxyParent());
//    if (proxyParentItem == m_dataSource->proxyRoot()){
//        return QModelIndex();
//    }
//    return createIndex(proxyParentItem->childNumber(), 0, proxyParentItem);
}

int LinkModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1;
}

int LinkModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }
//    BlockItem *proxyParentItem = blockFromQIndex(parent);
//    return proxyParentItem->childCount();
    return m_proxyPort->linkCount();
}

QVariant LinkModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    Link * linkItem = m_proxyPort->linkAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = linkItem->property(roleName.data());
    return name;
}

bool LinkModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        Link * linkItem = m_proxyPort->linkAt(index.row());
        switch (role) {
        case StartXRole:
//            if(linkItem->side() != value.toInt()){
//                linkItem->setSide(value.toInt());
//            }
            break;
        case StartYRole:
            break;
        case EndXRole:
            break;
        case EndYRole:
            break;
        }
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

Qt::ItemFlags LinkModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;

    return Qt::ItemIsEditable;
}

QHash<int, QByteArray> LinkModel::roleNames() const
{
    return m_roles;
}

Port *LinkModel::proxyPort() const
{
    return m_proxyPort;
}

void LinkModel::setProxyPort(Port *proxyPort)
{
    beginResetModel();
    if(m_proxyPort && m_signalConnected){
        m_proxyPort->disconnect(this);
    }

    m_proxyPort = proxyPort;

    connect(m_proxyPort,&Port::beginResetLinkModel,this,[=](){
        beginResetModel();
    });
    connect(m_proxyPort,&Port::endResetLinkModel,this,[=](){
        endResetModel();
    });
    connect(m_proxyPort,&Port::beginInsertLink,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_proxyPort,&Port::endInsertLink,this,[=](){
        endInsertRows();
    });
    connect(m_proxyPort,&Port::beginRemoveLink,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_proxyPort,&Port::endRemoveLink,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
