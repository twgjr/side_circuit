#include "diagrammodel.h"

DiagramModel::DiagramModel(QObject *parent) : QAbstractItemModel(parent),
    m_signalConnected(false),
    m_dataSource(nullptr)
{
    //qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;

    //set the basic roles for access of item properties in QML
    m_roles[ThisRole]="thisItem";
    m_roles[XposRole]="xPos";
    m_roles[YposRole]="yPos";
    m_roles[TypeRole]="type";
    m_roles[RotationRole]="rotation";
}

DiagramModel::~DiagramModel()
{
    //qDebug()<<"Block Model destroyed.";
}

QModelIndex DiagramModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }

    DiagramItem *item = m_dataSource->proxyRoot()->childItemAt(row);
    return createIndex(row, column, item);
}

QModelIndex DiagramModel::parent(const QModelIndex &index) const
{
    Q_UNUSED(index);
    return QModelIndex();
}

int DiagramModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }

    return m_dataSource->proxyRoot()->childItemCount();
}

int DiagramModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1; //columns not used
}

QVariant DiagramModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()){
        return QVariant();
    }

    DiagramItem * item = m_dataSource->proxyRoot()->childItemAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;
}

bool DiagramModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    bool somethingChanged = false;

    DiagramItem * item = m_dataSource->proxyRoot()->childItemAt(index.row());
    switch (role) {
    case XposRole:
        if(item->xPos() != value.toInt()){
            item->setXPos(value.toInt());
            somethingChanged = true;
        }
        break;
    case YposRole:
        if(item->yPos() != value.toInt()){
            item->setYPos(value.toInt());
            somethingChanged = true;
        }
        break;
    case TypeRole:
        if(item->type() != value.toInt()){
            item->setType(value.toInt());
            somethingChanged = true;
        }
        break;
    case RotationRole:
        if(item->rotation() != value.toInt()){
            item->setRotation(value.toInt());
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

Qt::ItemFlags DiagramModel::flags(const QModelIndex &index) const
{
    if (!index.isValid()){
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}

QHash<int, QByteArray> DiagramModel::roleNames() const
{
    return m_roles;
}

DataSource *DiagramModel::dataSource() const
{
    return m_dataSource;
}

void DiagramModel::setdataSource(DataSource *newDataSource)
{
    beginResetModel();
    if(m_dataSource && m_signalConnected){
        m_dataSource->disconnect(this);
    }

    m_dataSource = newDataSource;

    connect(m_dataSource,&DataSource::beginResetDiagramItems,this,[=](){
        beginResetModel();
    });
    connect(m_dataSource,&DataSource::endResetDiagramItems,this,[=](){
        endResetModel();
    });
    connect(m_dataSource,&DataSource::beginInsertDiagramItem,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endInsertDiagramItem,this,[=](){
        endInsertRows();
    });
    connect(m_dataSource,&DataSource::beginRemoveDiagramItem,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endRemoveDiagramItem,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
