#include "blockmodel.h"

BlockModel::BlockModel(QObject *parent) : QAbstractItemModel(parent)//,
  //connectedtoBlockDataSourceSignals(false)
{
    //setBlockDataSource(new BlockDataSource(this)); //connects the signals between datasource and model
    qDebug()<<"BlockModel object created.";
    m_root = new BlockItem(&m_context);

    m_roles[CategoryDataRole]="category";
    m_roles[IDDataRole]="id";
    m_roles[BlockXPositionRole]="blockXPosition";
    m_roles[BlockYPositionRole]="blockYPosition";
    m_roles[EquationRole]="equationString";
}

BlockModel::~BlockModel()
{
    delete m_root;
}

//counts the number of data items in the data source
int BlockModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0)
        return 0;
    BlockItem *parentItem = elementFromIndex(parent);
    return parentItem->count();
}

int BlockModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return 1;
}

//gets data from the data source
QVariant BlockModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    BlockItem * item = static_cast<BlockItem*>(index.internalPointer());
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;

    //    BlockItem * blockItem = m_blockDataSource->dataItems().at(index.row());
    //    if(role == CategoryDataRole){return blockItem->category();}
    //    if(role == IDDataRole){return blockItem->id();}
    //    if(role == BlockXPositionRole){return blockItem->blockXPosition();}
    //    if(role == BlockYPositionRole){return blockItem->blockYPosition();}
    //    if(role == EquationRole){return blockItem->equation()->getEquationString();}
    //    return QVariant();
}

bool BlockModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        BlockItem * blockItem = static_cast<BlockItem*>(index.internalPointer());
        switch (role) {
        case CategoryDataRole:
            if(blockItem->category() != value.toString()){
                blockItem->setCategory(value.toString());
            }
            break;
        case IDDataRole:
            if(blockItem->id() != value.toInt()){
                blockItem->setId(value.toInt());
            }
            break;
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
            }
            break;
        }
        //tell the QAbstractListModel that the data has changed
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

Qt::ItemFlags BlockModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;

    return QAbstractItemModel::flags(index);
}

QModelIndex BlockModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();
    BlockItem *parentItem = elementFromIndex(parent);
    BlockItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

QModelIndex BlockModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();
    BlockItem *childItem = static_cast<BlockItem*>(index.internalPointer());
    BlockItem *parentItem = static_cast<BlockItem *>(childItem->parentItem());
    if (parentItem == m_root)
        return QModelIndex();
    return createIndex(parentItem->pos(), 0, parentItem);
}

QHash<int, QByteArray> BlockModel::roleNames() const
{
    //refer to the value of roles[] to access in QML
    //for example, model.category would provide access
    //to the category role to qml properties or JS scripts
    return m_roles;
}

/*
BlockDataSource *BlockModel::blockDataSource() const
{
    return m_blockDataSource;
}

void BlockModel::setBlockDataSource(BlockDataSource *dataSource)
{
    // tell QAbstractListModel that the model is about to get
    // a new set of data from a new data source
    beginResetModel();

    //Disconnect any previously connected datasources
    if( m_blockDataSource && connectedtoBlockDataSourceSignals){
        m_blockDataSource->disconnect(this);
    }
    m_blockDataSource = dataSource;

    //Connect the datasource signals for adding and removing rows from the model to
    //QAbstractListModel
    connect(m_blockDataSource,&BlockDataSource::preItemAdded,this,[=](){
        const int index = m_blockDataSource->dataItems().count();
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_blockDataSource,&BlockDataSource::postItemAdded,this,[=](){
        endInsertRows();
    });
    connect(m_blockDataSource,&BlockDataSource::preItemRemoved,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_blockDataSource,&BlockDataSource::postItemRemoved,this,[=](){
        endRemoveRows();
    });

    endResetModel();
}
*/

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

void BlockModel::setRoles(QVariantList roles)
{
    //static int nextRole = Qt::UserRole + 1;
    static int nextRole = ModelRoles::EquationRole + 1;
    for(QVariant role : roles) {
        qDebug()<<role;
        m_roles.insert(nextRole, role.toByteArray());
        nextRole ++;
    }
    emit rolesChanged();
}

QModelIndex BlockModel::indexFromElement(BlockItem *item)
{
    QVector<int> positions;
    QModelIndex result;
    if(item) {
        do{
            int pos = item->pos();
            positions.append(pos);
            item = item->parentItem();
        } while(item != nullptr);

        for (int i = positions.size() - 2; i >= 0 ; i--)
            result = index(positions[i], 0, result);
    }
    return result;
}

bool BlockModel::insertElement(BlockItem *item, const QModelIndex &parent, int pos)
{
    BlockItem *parentElement = elementFromIndex(parent);
    if(pos >= parentElement->count())
        return false;
    if(pos < 0)
        pos = parentElement->count();
    beginInsertRows(parent, pos, pos);
    bool retValue = parentElement->insertItem(item, pos);
    endInsertRows();
    return retValue;
}

BlockItem *BlockModel::elementFromIndex(const QModelIndex &index) const
{
    if(index.isValid())
        return static_cast<BlockItem *>(index.internalPointer());
    return m_root;
}

//bool BlockModel::loadBlockItems(QVariant loadLocation){
//    return m_blockDataSource->loadBlockItems(loadLocation);
//}

//bool BlockModel::saveBlockItems(QVariant saveLocation){
//    return m_blockDataSource->saveBlockItems(saveLocation);
//}

void BlockModel::appendBlockItem(){
    BlockItem * item = new BlockItem(&m_context);
    insertElement(item);
}

//void BlockModel::clearBlockItems(){
//    m_blockDataSource->clearBlockItems();
//}

//void BlockModel::solveEquations(){
//    m_blockDataSource->solveEquations();
//}

//int BlockModel::maxBlockX(){
//    return m_blockDataSource->maxBlockX();
//}

//int BlockModel::maxBlockY(){
//    return m_blockDataSource->maxBlockY();
//}

//void BlockModel::removeLastBlockItem(){
//    m_blockDataSource->removeLastBlockItem();
//}
