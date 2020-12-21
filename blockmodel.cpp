#include "blockmodel.h"

BlockModel::BlockModel(QObject *parent) : QAbstractItemModel(parent),
    m_signalConnected(false)
{
    //qDebug()<<"BlockModel object created.";

    //set the basic roles for access of item properties in QML
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

//counts the number of data items in the data source
int BlockModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }
    BlockItem *proxyParentItem = blockFromQIndex(parent);
    return proxyParentItem->childCount();
}

int BlockModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1; //columns not used in this tree structure; 1 tells model data exists
}

//gets data from a block to display in QML model view
QVariant BlockModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()){
        return QVariant();
    }
    //BlockItem * item = static_cast<BlockItem*>(index.internalPointer());
    BlockItem * item = blockFromQIndex(index);
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;
}

//sets data from changing it in QML model view
bool BlockModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        BlockItem * blockItem = blockFromQIndex(index);
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
        //tell the QAbstractItemModel that the data has changed
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

BlockItem *BlockModel::blockFromQIndex(const QModelIndex &index) const
{
    if(index.isValid()){
        return static_cast<BlockItem *>(index.internalPointer());
    }
    return m_dataSource->proxyRoot();
}

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

Qt::ItemFlags BlockModel::flags(const QModelIndex &index) const
{
    if (!index.isValid()){
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}

QModelIndex BlockModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }
    BlockItem *proxyParentItem = blockFromQIndex(parent);
    BlockItem *proxyChildItem = proxyParentItem->child(row);
    if (proxyChildItem){
        return createIndex(row, column, proxyChildItem);
    } else {
        return QModelIndex();
    }
    return QModelIndex();
}


// gets QModelIndex of a child's parent
QModelIndex BlockModel::parent(const QModelIndex &index) const
{
    if (!index.isValid()){
        // the root index
        return QModelIndex();
    }
    BlockItem *proxyChildItem = static_cast<BlockItem*>(index.internalPointer());
    BlockItem *proxyParentItem = static_cast<BlockItem *>(proxyChildItem->proxyParent());
    if (proxyParentItem == m_dataSource->proxyRoot()){
        return QModelIndex();
    }
    return createIndex(proxyParentItem->childNumber(), 0, proxyParentItem);
}

QHash<int, QByteArray> BlockModel::roleNames() const {return m_roles;}

QVariantList BlockModel::roles() const
{
    QVariantList list;
    QHashIterator<int, QByteArray> i(m_roles);
    while (i.hasNext()) {
        i.next();
        list.append(i.value());
        qDebug()<<i.value();
    }
    return list;
}

//May not be necessary unless need to create roles dynamically
void BlockModel::setRoles(QVariantList roles)
{
    static int nextRole = EquationRole + 1;
    for(QVariant role : roles) {
        qDebug()<<role;
        m_roles.insert(nextRole, role.toByteArray());
        nextRole ++;
    }
    emit rolesChanged();
}

QModelIndex BlockModel::qIndexOfBlock(BlockItem *item)
{
    QVector<int> positions;
    QModelIndex result;
    if(item) {
        do{
            int pos = item->proxyChildNumber();
            positions.append(pos);
            item = item->proxyParent();
        } while(item != nullptr);

        for (int i = positions.size() - 2; i >= 0 ; i--){
            result = index(positions[i], 0, result);
        }
    }
    return result;
}
