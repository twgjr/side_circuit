#include "portmodel.h"

PortModel::PortModel(QObject *parent)
    : QAbstractListModel(parent),
      m_signalConnected(false)//,
      //m_blockParent(nullptr)
{
    //set the basic roles for access of item properties in QML
    m_roles[PortModel::SideRole]="side";
    m_roles[PortModel::PositionRole]="position";
    m_roles[PortModel::NameRole]="name";

    qDebug()<<"Port model created";
}

int PortModel::rowCount(const QModelIndex &parent) const
{
    // For list models only the root node (an invalid parent) should return the list's size. For all
    // other (valid) parents, rowCount() should return 0 so that it does not become a tree model.
    if (parent.isValid())
        return 0;

    // FIXME: Implement me!
    return m_blockDataSource->portCount();
}

QVariant PortModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    Port * portItem = m_blockDataSource->ports()[index.row()];
    QByteArray roleName = m_roles[role];
    QVariant name = portItem->property(roleName.data());
    return name;
}

bool PortModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
            //don't do anything if the data being submitted did not change
            Port * portItem = m_blockDataSource->ports()[index.row()];
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

    return Qt::ItemIsEditable; // FIXME: Implement me!
}

QHash<int, QByteArray> PortModel::roleNames() const
{
    return m_roles;
}

BlockItem *PortModel::blockDataSource() const
{
    return m_blockDataSource;
}

void PortModel::setBlockDataSource(BlockItem *newBlockDataSource)
{
//    if (m_blockDataSource == blockDataSource)
//        return;

//    m_blockDataSource = blockDataSource;
//    emit blockDataSourceChanged(m_blockDataSource);
    beginResetModel();
    if(m_blockDataSource && m_signalConnected){
        m_blockDataSource->disconnect(this);
    }

    m_blockDataSource = newBlockDataSource;

    connect(m_blockDataSource,&BlockItem::beginResetPortModel,this,[=](){
        beginResetModel();
    });
    connect(m_blockDataSource,&BlockItem::endResetPortModel,this,[=](){
        endResetModel();
    });
    connect(m_blockDataSource,&BlockItem::beginInsertPort,this,[=](){
        const int index = m_blockDataSource->portCount();
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_blockDataSource,&BlockItem::endInsertPort,this,[=](){
        endInsertRows();
    });
    connect(m_blockDataSource,&BlockItem::beginRemovePort,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_blockDataSource,&BlockItem::endRemovePort,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
/*
void PortModel::addPort(int side, int position)
{
    Port * portItem = new Port();
    portItem->setBlockParent(m_blockParent);
    portItem->setParent(m_blockParent);
    portItem->setSide(side);
    portItem->setPosition(position);
    const int index = m_ports.size();
    beginInsertRows(QModelIndex(),index,index);
    m_ports.append(portItem);
    endInsertRows();
}
*/
/*
BlockItem *PortModel::blockParent() const
{
    return m_blockParent;
}
*/
/*
void PortModel::setBlockParent(BlockItem *blockParent)
{
    if (m_blockParent == blockParent)
        qDebug()<<"Port model parent: "<<m_blockParent;
        return;

    m_blockParent = blockParent;
    qDebug()<<"Port model parent: "<<m_blockParent;
}
*/
