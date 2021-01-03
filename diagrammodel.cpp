#include "diagrammodel.h"

DiagramModel::DiagramModel(QObject *parent) : QAbstractItemModel(parent),
    m_signalConnected(false)
{
    //qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;

    //set the basic roles for access of item properties in QML
    m_roles[ProxyRoot]="proxyRoot";
    m_roles[ThisBlock]="thisBlock";
    m_roles[DescriptionDataRole]="description";
    m_roles[IDDataRole]="id";
    m_roles[BlockXPositionRole]="blockXPosition";
    m_roles[BlockYPositionRole]="blockYPosition";
    m_roles[blockHeightRole]="blockHeight";
    m_roles[blockWidthRole]="blockWidth";
    m_roles[EquationRole]="equationString";
    m_roles[ThisRole]="thisBlock";
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
    //index all blocks, then equations
    Block *childBlockItem = m_dataSource->proxyRoot()->childBlockAt(row);
    if (childBlockItem){
        return createIndex(row, column, childBlockItem);
    }
    Equation *childEquationItem = m_dataSource->proxyRoot()->childEquationAt(row);
    if (childEquationItem){
        return createIndex(row, column, childEquationItem);
    }
    return QModelIndex();
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
    //return m_dataSource->proxyRoot()->childBlockCount();
    return m_dataSource->proxyRoot()->diagramItemCount();
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

    if(index.row() < m_dataSource->proxyRoot()->childBlockCount()){
        Block * item = m_dataSource->proxyRoot()->childBlockAt(index.row());
        QByteArray roleName = m_roles[role];
        QVariant name = item->property(roleName.data());
        return name;
    } else {
        Equation * item = m_dataSource->proxyRoot()->childEquationAt(index.row());
        QByteArray roleName = m_roles[role+BlockRolesEnd-1];
        QVariant name = item->property(roleName.data());
        return name;
    }

    return QVariant();
}

bool DiagramModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    bool somethingChanged = false;

    if(index.row() < m_dataSource->proxyRoot()->childBlockCount()){
        Block * blockItem = m_dataSource->proxyRoot()->childBlockAt(index.row());
        switch (role) {
        case DescriptionDataRole:
            if(blockItem->description() != value.toString()){
                blockItem->setDescription(value.toString());
            }
            break;
        case IDDataRole: break;
        case BlockXPositionRole:
            if(blockItem->blockXPosition() != value.toInt()){
                blockItem->setBlockXPosition(value.toInt());
            }
            break;
        case BlockYPositionRole:
            if(blockItem->blockYPosition() != value.toInt()){
                blockItem->setBlockYPosition(value.toInt());
            }
            break;
        }
    } else {
        Equation * equationItem = m_dataSource->proxyRoot()->childEquationAt(index.row());

        switch (role-(BlockRolesEnd-1)) {
        case EqXPosRole:
            if(equationItem->eqXPos() != value.toInt()){
                equationItem->setEqXPos(value.toInt());
            }
            break;
        case EqYPosRole:
            if(equationItem->eqYPos() != value.toInt()){
                equationItem->setEqYPos(value.toInt());
            }
            break;
        case EquationRole:
            if(equationItem->getEquationString() != value.toString()){
                equationItem->setEquationString(value.toString());
                equationItem->eqStrToExpr();
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

Qt::ItemFlags DiagramModel::flags(const QModelIndex &index) const
{
    if (!index.isValid()){
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}

QHash<int, QByteArray> DiagramModel::roleNames() const {return m_roles;}

DataSource *DiagramModel::dataSource() const
{
    return m_dataSource;
}

void DiagramModel::setdataSource(DataSource *newBlockDataSource)
{
    beginResetModel();
    if(m_dataSource && m_signalConnected){
        m_dataSource->disconnect(this);
    }

    m_dataSource = newBlockDataSource;

    connect(m_dataSource,&DataSource::beginResetDiagram,this,[=](){
        beginResetModel();
    });
    connect(m_dataSource,&DataSource::endResetDiagram,this,[=](){
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
