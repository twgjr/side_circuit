#include "portmodel.h"

PortModel::PortModel(QObject *parent)
    : QAbstractListModel(parent),
      m_signalConnected(false),
      m_dsChildBlock(nullptr)
{
    //set the basic roles for access of item properties in QML
    m_roles[SideRole]="side";
    m_roles[PositionRole]="position";
    m_roles[NameRole]="name";

    qDebug()<<"Created: "<<this;
}

PortModel::~PortModel()
{
    qDebug()<<"Deleted: "<<this;
}

int PortModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;

    return m_dsChildBlock->dsChildBlock()->portCount();
}

QVariant PortModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    Port * portItem = m_dsChildBlock->dsChildBlock()->ports()[index.row()];
    QByteArray roleName = m_roles[role];
    QVariant name = portItem->property(roleName.data());
    return name;
}

bool PortModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        Port * portItem = m_dsChildBlock->dsChildBlock()->ports()[index.row()];
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

DSChildBlock *PortModel::dsChildBlock()
{
    return m_dsChildBlock;
}

void PortModel::setdsChildBlock(DSChildBlock* proxyBlock)
{
    beginResetModel();
    if(m_dsChildBlock && m_signalConnected){
        m_dsChildBlock->disconnect(this);
    }

    m_dsChildBlock = proxyBlock;

    connect(m_dsChildBlock,&DSChildBlock::beginResetPortModel,this,[=](){
        beginResetModel();
    });
    connect(m_dsChildBlock,&DSChildBlock::endResetPortModel,this,[=](){
        endResetModel();
    });
    connect(m_dsChildBlock,&DSChildBlock::beginInsertPort,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_dsChildBlock,&DSChildBlock::endInsertPort,this,[=](){
        endInsertRows();
    });
    connect(m_dsChildBlock,&DSChildBlock::beginRemovePort,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_dsChildBlock,&DSChildBlock::endRemovePort,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
