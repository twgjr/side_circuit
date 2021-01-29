#include "linkmodel.h"

LinkModel::LinkModel(QObject *parent) : QAbstractItemModel(parent),
    m_signalConnected(false)
{
    //set the basic roles for access of item properties in QML
    m_roles[ThisLinkRole]="thisLink";
    m_roles[LastPointRole]="lastPoint";
    m_roles[PortConnectedRole]="portConnected";

    //qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

LinkModel::~LinkModel()
{
    //qDebug()<<"Deleted: "<<this;
}

QModelIndex LinkModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }
    Link * linkItem = m_proxyPort->linkAt(row);
    if (linkItem){
        return createIndex(row, column, linkItem);
    }
    return QModelIndex();
}

QModelIndex LinkModel::parent(const QModelIndex &index) const
{
    Q_UNUSED(index);
    return QModelIndex();
}

int LinkModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }
    return m_proxyPort->linkCount();
}

int LinkModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1;
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
    Link * item = m_proxyPort->linkAt(index.row());
    bool somethingChanged = false;
    switch (role) {
    case LastPointRole:
        if(item->lastPoint() != value.toPoint()){
            item->setLastPoint(value.toPoint());
            somethingChanged = true;
        }
        break;
    }
    if(somethingChanged){
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

    connect(m_proxyPort,&Port::beginResetPort,this,[=](){
        beginResetModel();
    });
    connect(m_proxyPort,&Port::endResetPort,this,[=](){
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
