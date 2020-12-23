#include "portmodel.h"

PortModel::PortModel(QObject *parent): QAbstractItemModel(parent),
      m_signalConnected(false)
{
    //set the basic roles for access of item properties in QML
    m_roles[SideRole]="side";
    m_roles[PositionRole]="position";
    m_roles[NameRole]="name";

    qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

PortModel::~PortModel()
{
    qDebug()<<"Deleted: "<<this;
}

QModelIndex PortModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }
//    BlockItem *proxyParentItem = blockFromQIndex(parent);
//    BlockItem *proxyChildItem = proxyParentItem->child(row);
    Port * portItem = m_proxyChildBlock->portAt(row);
    if (portItem){
        return createIndex(row, column, portItem);
    }

    return QModelIndex();
}

QModelIndex PortModel::parent(const QModelIndex &index) const
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

int PortModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1;
}

int PortModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }
//    BlockItem *proxyParentItem = blockFromQIndex(parent);
//    return proxyParentItem->childCount();
    return m_proxyChildBlock->portCount();
}

QVariant PortModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    Port * portItem = m_proxyChildBlock->portAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = portItem->property(roleName.data());
    return name;
}

bool PortModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        Port * portItem = m_proxyChildBlock->portAt(index.row());
        switch (role) {
        case SideRole:
            if(portItem->side() != value.toInt()){
                portItem->setSide(value.toInt());
            }
            break;
        case PositionRole:
            if(portItem->position() != value.toInt()){
                portItem->setPosition(value.toInt());
            }
            break;
        case NameRole:
            if(portItem->name() != value.toString()){
                portItem->setName(value.toString());
            }
            break;
        }
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

Qt::ItemFlags PortModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;

    return Qt::ItemIsEditable;
}

QHash<int, QByteArray> PortModel::roleNames() const
{
    return m_roles;
}

BlockItem *PortModel::proxyChildBlock() const
{
    return m_proxyChildBlock;
}

void PortModel::setProxyChildBlock(BlockItem *proxyChildBlock)
{
    beginResetModel();
    if(m_proxyChildBlock && m_signalConnected){
        m_proxyChildBlock->disconnect(this);
    }
    m_proxyChildBlock = proxyChildBlock;

    connect(m_proxyChildBlock,&BlockItem::beginResetPortModel,this,[=](){
        beginResetModel();
    });
    connect(m_proxyChildBlock,&BlockItem::endResetPortModel,this,[=](){
        endResetModel();
    });
    connect(m_proxyChildBlock,&BlockItem::beginInsertPort,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_proxyChildBlock,&BlockItem::endInsertPort,this,[=](){
        endInsertRows();
    });
    connect(m_proxyChildBlock,&BlockItem::beginRemovePort,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_proxyChildBlock,&BlockItem::endRemovePort,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
