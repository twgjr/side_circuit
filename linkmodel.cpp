#include "linkmodel.h"

LinkModel::LinkModel(QObject *parent) : QAbstractListModel(parent),
    m_signalConnected(false),
    m_dsPort(nullptr)
{
    //set the basic roles for access of item properties in QML
    m_roles[StartXRole]="startX";
    m_roles[StartYRole]="startY";
    m_roles[EndXRole]="endX";
    m_roles[EndYRole]="endY";

    qDebug()<<"Created: "<<this;
}

LinkModel::~LinkModel()
{
    qDebug()<<"Deleted: "<<this;
}

int LinkModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;

    return m_dsPort->dsPort()->linkCount();
}

QVariant LinkModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    Link * linkItem = m_dsPort->dsPort()->links()[index.row()];
    QByteArray roleName = m_roles[role];
    QVariant name = linkItem->property(roleName.data());
    return name;
}

bool LinkModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        Link * linkItem = m_dsPort->dsPort()->links()[index.row()];
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

DSPort *LinkModel::dsPort()
{
    return m_dsPort;
}

void LinkModel::setdsPort(DSPort *proxyPort)
{
    beginResetModel();
    if(m_dsPort && m_signalConnected){
        m_dsPort->disconnect(this);
    }

    m_dsPort = proxyPort;

    connect(m_dsPort,&DSPort::beginResetLinkModel,this,[=](){
        beginResetModel();
    });
    connect(m_dsPort,&DSPort::endResetLinkModel,this,[=](){
        endResetModel();
    });
    connect(m_dsPort,&DSPort::beginInsertLink,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_dsPort,&DSPort::endInsertLink,this,[=](){
        endInsertRows();
    });
    connect(m_dsPort,&DSPort::beginRemoveLink,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_dsPort,&DSPort::endRemoveLink,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
