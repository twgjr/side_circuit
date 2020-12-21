#include "portmodel.h"

PortModel::PortModel(QObject *parent)
    : QAbstractListModel(parent),
      m_signalConnected(false)
{
    //set the basic roles for access of item properties in QML
    m_roles[PortModel::SideRole]="side";
    m_roles[PortModel::PositionRole]="position";
    m_roles[PortModel::NameRole]="name";

    //qDebug()<<"Port model created";
}

PortModel::~PortModel()
{
    //qDebug()<<"Port model destroyed";
}

int PortModel::rowCount(const QModelIndex &parent) const
{
    // For list models only the root node (an invalid parent) should return the list's size. For all
    // other (valid) parents, rowCount() should return 0 so that it does not become a tree model.
    if (parent.isValid())
        return 0;

    return m_proxyPorts->parentBlock()->portCount();
}

QVariant PortModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    Port * portItem = m_proxyPorts->parentBlock()->ports()[index.row()];
    QByteArray roleName = m_roles[role];
    QVariant name = portItem->property(roleName.data());
    return name;
}

bool PortModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        Port * portItem = m_proxyPorts->parentBlock()->ports()[index.row()];
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

ProxyPorts *PortModel::proxyPorts()
{
    return m_proxyPorts;
}

void PortModel::setProxyPorts(ProxyPorts* portDataSource)
{
    beginResetModel();
    if(m_proxyPorts && m_signalConnected){
        m_proxyPorts->disconnect(this);
    }

    m_proxyPorts = portDataSource;

    connect(m_proxyPorts,&ProxyPorts::beginResetPortModel,this,[=](){
        beginResetModel();
    });
    connect(m_proxyPorts,&ProxyPorts::endResetPortModel,this,[=](){
        endResetModel();
    });
    connect(m_proxyPorts,&ProxyPorts::beginInsertPort,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_proxyPorts,&ProxyPorts::endInsertPort,this,[=](){
        endInsertRows();
    });
    connect(m_proxyPorts,&ProxyPorts::beginRemovePort,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_proxyPorts,&ProxyPorts::endRemovePort,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
