#include "portmodel.h"

PortModel::PortModel(QObject *parent): QAbstractItemModel(parent),
    m_signalConnected(false),
    m_parentItem(nullptr)
{
    //set the basic roles for access of item properties in QML
    m_roles[ThisPort]="thisPort";
    m_roles[SideRole]="side";
    m_roles[PositionRole]="position";
    m_roles[NameRole]="name";
    m_roles[AbsPointRole]="absPoint";
    m_roles[LinkIsValidRole]="linkIsValid";

    //qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

PortModel::~PortModel()
{
    //qDebug()<<"Deleted: "<<this;
}

QModelIndex PortModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }

    Port * item = m_parentItem->portAt(row);
    return createIndex(row, column, item);
}

QModelIndex PortModel::parent(const QModelIndex &index) const
{
    Q_UNUSED(index)
    return QModelIndex();
}

int PortModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }

    return m_parentItem->portCount();
}

int PortModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1;
}

QVariant PortModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()){
        return QVariant();
    }

    Port * item = m_parentItem->portAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;
}

bool PortModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    bool somethingChanged = false;

    Port * item = m_parentItem->portAt(index.row());
    switch (role) {
    case AbsPointRole:
        if(item->absPoint() != value.toPoint()){
            item->setAbsPoint(value.toPoint());
            somethingChanged = true;
        }
        break;
    case LinkIsValidRole:
        if(item->linkIsValid() != value.toBool()){
            item->setLinkIsValid(value.toBool());
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

DiagramItem *PortModel::parentItem() const
{
    return m_parentItem;
}

void PortModel::setParentItem(DiagramItem *parentItem)
{
    beginResetModel();
    if(m_parentItem && m_signalConnected){
        m_parentItem->disconnect(this);
    }
    m_parentItem = parentItem;

    connect(m_parentItem,&DiagramItem::beginResetPorts,this,[=](){
        beginResetModel();
    });
    connect(m_parentItem,&DiagramItem::endResetPorts,this,[=](){
        endResetModel();
    });
    connect(m_parentItem,&DiagramItem::beginInsertPort,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_parentItem,&DiagramItem::endInsertPort,this,[=](){
        endInsertRows();
    });
    connect(m_parentItem,&DiagramItem::beginRemovePort,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_parentItem,&DiagramItem::endRemovePort,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
