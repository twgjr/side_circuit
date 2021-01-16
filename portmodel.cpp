#include "portmodel.h"

PortModel::PortModel(QObject *parent): QAbstractItemModel(parent),
    m_signalConnected(false),
    m_proxyChildBlock(nullptr),
    m_proxyElement(nullptr)
{
    //set the basic roles for access of item properties in QML
    m_roles[ThisPort]="thisPort";
    m_roles[SideRole]="side";
    m_roles[PositionRole]="position";
    m_roles[NameRole]="name";
    m_roles[State]="state";

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
    if(m_proxyChildBlock){
        Port * portItem = m_proxyChildBlock->portAt(row);

        if (portItem){
            return createIndex(row, column, portItem);
        }
    }

    if(m_proxyElement){
        Port * portItem = m_proxyElement->portAt(row);

        if (portItem){
            return createIndex(row, column, portItem);
        }
    }

    return QModelIndex();
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

    if(m_proxyChildBlock){
        return m_proxyChildBlock->portCount();
    }

    if(m_proxyElement){
        return m_proxyElement->portCount();
    }

    return 0;
}

int PortModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1;
}

QVariant PortModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if(m_proxyChildBlock){
    Port * portItem = m_proxyChildBlock->portAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = portItem->property(roleName.data());
    return name;
    }

    if(m_proxyElement){
    Port * portItem = m_proxyElement->portAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = portItem->property(roleName.data());
    return name;
    }
    return QVariant();
}

bool PortModel::setData(const QModelIndex &index, const QVariant &value, int role)
{

    bool somethingChanged = false;
    Port * portItem;
    if(m_proxyChildBlock){
        portItem = m_proxyChildBlock->portAt(index.row());
    }
    if(m_proxyElement){
        portItem = m_proxyElement->portAt(index.row());
    }
    if(portItem){
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
        case State:
            if(portItem->name() != value.toString()){
                portItem->setName(value.toString());
            }
            break;
        }
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

Block *PortModel::proxyChildBlock() const
{
    return m_proxyChildBlock;
}

void PortModel::setProxyChildBlock(Block *proxyChildBlock)
{
    m_proxyElement = nullptr;
    beginResetModel();
    if(m_proxyChildBlock && m_signalConnected){
        m_proxyChildBlock->disconnect(this);
    }
    m_proxyChildBlock = proxyChildBlock;

    connect(m_proxyChildBlock,&Block::beginResetBlock,this,[=](){
        beginResetModel();
    });
    connect(m_proxyChildBlock,&Block::endResetBlock,this,[=](){
        endResetModel();
    });
    connect(m_proxyChildBlock,&Block::beginInsertPort,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_proxyChildBlock,&Block::endInsertPort,this,[=](){
        endInsertRows();
    });
    connect(m_proxyChildBlock,&Block::beginRemovePort,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_proxyChildBlock,&Block::endRemovePort,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}

Element *PortModel::proxyElement() const
{
    return m_proxyElement;
}

void PortModel::setProxyElement(Element *proxyElement)
{
    m_proxyChildBlock = nullptr;
    beginResetModel();
    if(m_proxyElement && m_signalConnected){
        m_proxyElement->disconnect(this);
    }
    m_proxyElement = proxyElement;

    connect(m_proxyElement,&Element::beginResetElement,this,[=](){
        beginResetModel();
    });
    connect(m_proxyElement,&Element::endResetElement,this,[=](){
        endResetModel();
    });
    connect(m_proxyElement,&Element::beginInsertPort,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_proxyElement,&Element::endInsertPort,this,[=](){
        endInsertRows();
    });
    connect(m_proxyElement,&Element::beginRemovePort,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_proxyElement,&Element::endRemovePort,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
