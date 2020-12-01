#include "blockmodel.h"

BlockModel::BlockModel(QObject *parent) : QAbstractListModel(parent),
    connectedtoBlockDataSourceSignals(false)
{
    setBlockDataSource(new BlockDataSource(this)); //connects the signals between datasource and model
    qDebug()<<"BlockModel object created.";
}

//counts the number of data items in the data source
int BlockModel::rowCount(const QModelIndex &parent) const
{
    // For list models only the root node (an invalid parent) should return the list's size. For all
    // other (valid) parents, rowCount() should return 0 so that it does not become a tree model.
    if (parent.isValid())
        return 0;

    return m_blockDataSource->dataItems().count();
}

//gets data from the data source
QVariant BlockModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    BlockItem * blockItem = m_blockDataSource->dataItems().at(index.row());

    if(role == CategoryDataRole){
        return blockItem->category();
    }
    if(role == IDDataRole){
        return blockItem->id();
    }
    if(role == BlockXPositionRole){
        return blockItem->blockXPosition();
    }
    if(role == BlockYPositionRole){
        return blockItem->blockYPosition();
    }
    if(role == EquationRole){
        return blockItem->equation()->getEquationString();
    }

    return QVariant();
}

bool BlockModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        BlockItem * blockItem = m_blockDataSource->dataItems().at(index.row());
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

    return Qt::ItemIsEditable;
}

QHash<int, QByteArray> BlockModel::roleNames() const
{
    //refer to the value of roles[] to access in QML
    //for example, model.category would provide access
    //to the category role to qml properties or JS scripts
    QHash<int,QByteArray> roles;
    roles[CategoryDataRole]="category";
    roles[IDDataRole]="id";
    roles[BlockXPositionRole]="blockXPosition";
    roles[BlockYPositionRole]="blockYPosition";
    roles[EquationRole]="equation";

    return roles;
}

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
