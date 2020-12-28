#include "blockmodel.h"

BlockModel::BlockModel(QObject *parent) : QAbstractItemModel(parent),
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

BlockModel::~BlockModel()
{
    //qDebug()<<"Block Model destroyed.";
}

QModelIndex BlockModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }
    BlockItem *proxyChildItem = m_dataSource->proxyRoot()->childBlock(row);
    if (proxyChildItem){
        return createIndex(row, column, proxyChildItem);
    }
    return QModelIndex();
}

QModelIndex BlockModel::parent(const QModelIndex &index) const
{
    Q_UNUSED(index);
    return QModelIndex();
}

int BlockModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }
    return m_dataSource->proxyRoot()->childBlockCount();
}

int BlockModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1; //columns not used
}

QVariant BlockModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()){
        return QVariant();
    }
    BlockItem * item = m_dataSource->proxyRoot()->childBlock(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;
}

bool BlockModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    BlockItem * blockItem = m_dataSource->proxyRoot()->childBlock(index.row());
    bool somethingChanged = false;
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
    case EquationRole:
        if(blockItem->equation()->getEquationString() != value.toString()){
            blockItem->equation()->setEquationString(value.toString());
            blockItem->equation()->eqStrToExpr();
        }
        break;
    }
    if(somethingChanged){
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

Qt::ItemFlags BlockModel::flags(const QModelIndex &index) const
{
    if (!index.isValid()){
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}

QHash<int, QByteArray> BlockModel::roleNames() const {return m_roles;}

DataSource *BlockModel::dataSource() const
{
    return m_dataSource;
}

void BlockModel::setdataSource(DataSource *newBlockDataSource)
{
    beginResetModel();
    if(m_dataSource && m_signalConnected){
        m_dataSource->disconnect(this);
    }

    m_dataSource = newBlockDataSource;

    connect(m_dataSource,&DataSource::beginResetBlockModel,this,[=](){
        beginResetModel();
    });
    connect(m_dataSource,&DataSource::endResetBlockModel,this,[=](){
        endResetModel();
    });
    connect(m_dataSource,&DataSource::beginInsertBlock,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endInsertBlock,this,[=](){
        endInsertRows();
    });
    connect(m_dataSource,&DataSource::beginRemoveBlock,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endRemoveBlock,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
