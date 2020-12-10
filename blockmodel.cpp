#include "blockmodel.h"

BlockModel::BlockModel(QObject *parent) : QAbstractItemModel(parent)
{
    qDebug()<<"BlockModel object created.";

    m_root = new BlockItem(&m_context,nullptr,this);  // real root is empty
    m_proxyRoot = new BlockItem(&m_context,nullptr,this);
    newProxyRoot(m_root);  //clone real root into proxy root

    //set the basic roles for access of item properties in QML
    m_roles[BlockItem::CategoryDataRole]="category";
    m_roles[BlockItem::IDDataRole]="id";
    m_roles[BlockItem::BlockXPositionRole]="blockXPosition";
    m_roles[BlockItem::BlockYPositionRole]="blockYPosition";
    m_roles[BlockItem::EquationRole]="equationString";
}

BlockModel::~BlockModel()
{
    delete m_root;
    delete m_proxyRoot;
    qDebug()<<"Block Model destroyed.";

}

//counts the number of data items in the data source
int BlockModel::rowCount(const QModelIndex &parent) const
{
    const BlockItem *parentItem = getItemFromQIndex(parent);

    return parentItem ? parentItem->childCount() : 0;
}

int BlockModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1; //columns not used in this tree structure, dummy 1 to tell it data exists
}

//gets data from a block to display in QML model view
QVariant BlockModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    //BlockItem * item = static_cast<BlockItem*>(index.internalPointer());
    BlockItem * item = getItemFromQIndex(index);
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;
}

//sets data from changing it in QML model view
bool BlockModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (data(index, role) != value) {
        //don't do anything if the data being submitted did not change
        BlockItem * proxyBlockItem = getItemFromQIndex(index);
        BlockItem * realBlockItem = proxyBlockItem->realModelPointer();
        switch (role) {
        case BlockItem::CategoryDataRole:
            if(proxyBlockItem->category() != value.toString()){
                proxyBlockItem->setCategory(value.toString());
                realBlockItem->setCategory(value.toString());
            }
            break;
        case BlockItem::IDDataRole: break;
        case BlockItem::BlockXPositionRole:
            if(proxyBlockItem->blockXPosition() != value.toInt()){
                proxyBlockItem->setBlockXPosition(value.toInt());
                realBlockItem->setBlockXPosition(value.toInt());
            }
            break;
        case BlockItem::BlockYPositionRole:
            if(proxyBlockItem->blockYPosition() != value.toInt()){
                proxyBlockItem->setBlockYPosition(value.toInt());
                realBlockItem->setBlockYPosition(value.toInt());
            }
            break;
        case BlockItem::EquationRole:
            if(proxyBlockItem->equation()->getEquationString() != value.toString()){
                proxyBlockItem->equation()->setEquationString(value.toString());
                realBlockItem->equation()->setEquationString(value.toString());
                realBlockItem->equation()->eqStrToExpr();
            }
            break;
        }
        //tell the QAbstractItemModel that the data has changed
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

bool BlockModel::insertRows(int position, int rows, const QModelIndex &parent)
{
    BlockItem *parentItem = getItemFromQIndex(parent);
    if (!parentItem)
        return false;
    beginInsertRows(parent, position, position + rows - 1);
    const bool success = parentItem->insertChildren(position,
                                                    rows,
                                                    m_root->columnCount());
    endInsertRows();
    return success;
}

bool BlockModel::removeRows(int position, int rows, const QModelIndex &parent)
{
    BlockItem *parentItem = getItemFromQIndex(parent);
    if (!parentItem)
        return false;
    beginRemoveRows(parent, position, position + rows - 1);
    const bool success = parentItem->removeChildren(position, rows);
    endRemoveRows();
    return success;
}

BlockItem *BlockModel::getItemFromQIndex(const QModelIndex &index) const
{
    if (index.isValid()) {
        BlockItem *item = static_cast<BlockItem*>(index.internalPointer());
        return item;
    }
    return m_proxyRoot;
}

Qt::ItemFlags BlockModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;

    return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}

QModelIndex BlockModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    BlockItem *parentItem = getItemFromQIndex(parent);
    if (!parentItem)
        return QModelIndex();

    BlockItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);

    return QModelIndex();
}

QModelIndex BlockModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    BlockItem *childItem = getItemFromQIndex(index);
    BlockItem *parentItem = childItem ? childItem->parentItem() : nullptr;

    if (parentItem == /*m_root*/m_proxyRoot || !parentItem)
        return QModelIndex();

    return createIndex(parentItem->childNumber(), 0, parentItem);
}

QHash<int, QByteArray> BlockModel::roleNames() const
{
    return m_roles;
}

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
    //static int nextRole = Qt::UserRole + 1;
    static int nextRole = BlockItem::ModelRoles::EquationRole + 1;
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
            int pos = item->childNumber();
            positions.append(pos);
            item = item->parentItem();
        } while(item != nullptr);

        for (int i = positions.size() - 2; i >= 0 ; i--)
            result = index(positions[i], 0, result);
    }
    return result;
}

void BlockModel::appendBlock()
{
    //append new block to real model
    BlockItem *realChild = new BlockItem(&m_context,nullptr,this);
    BlockItem *realParentItem = m_proxyRoot->realModelPointer();
    realParentItem->appendChild(realChild);

    //append copy of new block (row) to proxy model and update the gui tree model
    int nextAvailableProxyPos = m_proxyRoot->childCount();
    QModelIndex proxyParentIndex = qIndexOfBlock(m_proxyRoot);
    beginInsertRows(proxyParentIndex, nextAvailableProxyPos, nextAvailableProxyPos);
    newProxyRoot(realParentItem);  // reset the proxy model
    endInsertRows();
}

void BlockModel::downLevel(int modelIndex)
{
    //set child clicked as new proxy root
    beginResetModel();
    newProxyRoot(m_proxyRoot->realModelPointer()->child(modelIndex));
    endResetModel();
}

void BlockModel::upLevel()
{
    //retrieve real parent and copy to proxy unless already at real root
    if(m_proxyRoot->realModelPointer()->parentItem()!=nullptr){
        // parent is valid, cannot go higher than root
        beginResetModel();
        // set parent of proxy root as new root
        newProxyRoot(m_proxyRoot->realModelPointer()->parentItem());
        endResetModel();
    }
}

void BlockModel::printDebugTree(BlockItem *parentItem, int depth)
{
    //iterate through all children and load equations then recurse to children
    if(parentItem->parentItem() == nullptr){
        qDebug() << "ROOT at level:"<<depth;
    }

    for (int i = 0 ; i < parentItem->childCount() ; i++) {
        if( parentItem->child(i)->childCount() == 0 ){
            //is a leaf, then print and return, else continue to traverse the tree
            qDebug()<<"equation: "<<parentItem->child(i)->equation()->getEquationString()
                   << "at depth: "<<depth+1;
        } else{
            printDebugTree(parentItem->child(i),depth+1);
        }
    }
}

void BlockModel::printBlock(int modelIndex)
{
    qDebug()<<"ID: " << m_proxyRoot->child(modelIndex)->realModelPointer()->id();
    qDebug()<<"Category: " << m_proxyRoot->child(modelIndex)->realModelPointer()->category();
    qDebug()<<"Position: " << m_proxyRoot->child(modelIndex)->realModelPointer()->blockXPosition()
                            << " x "
                            << m_proxyRoot->child(modelIndex)->realModelPointer()->blockYPosition();
    qDebug()<<"Equation: " << m_proxyRoot->child(modelIndex)->realModelPointer()->equationString();
}

int BlockModel::distanceFromRoot() const {

    BlockItem * realItem = m_proxyRoot->realModelPointer();

    int count = 0;
    if(realItem->parentItem()==nullptr){
        return count;
    } else {
        realItem = realItem->parentItem();
        count+=1;
        while(realItem->parentItem()!=nullptr){
            realItem = realItem->parentItem();
            count+=1;
        }
        return count;
    }
}

int BlockModel::numChildren(int modelIndex)
{
    return m_proxyRoot->child(modelIndex)->realModelPointer()->childCount();
}

BlockItem *BlockModel::blockFromQIndex(const QModelIndex &index) const
{
    if(index.isValid())
        return static_cast<BlockItem *>(index.internalPointer());
    return m_root;
}

void BlockModel::newProxyRoot(BlockItem *newRealModelPointer)
{
    delete m_proxyRoot; // destroy old proxy root

    // create a new empty proxy root pointer
    BlockItem* newProxyItem = new BlockItem(&m_context,nullptr,this);
    m_proxyRoot = newProxyItem;

    newProxyItem->setCategory(newRealModelPointer->category());
    newProxyItem->setBlockXPosition(newRealModelPointer->blockXPosition());
    newProxyItem->setBlockYPosition(newRealModelPointer->blockYPosition());
    newProxyItem->equation()->setEquationExpression(newRealModelPointer->equation()->getEquationExpression());
    //copy original true parent pointer for reference to real model
    newProxyItem->setRealModelPointer(newRealModelPointer);

    //copy new child data
    for ( int i = 0 ; i < newRealModelPointer->childCount() ; i++ ) {
        BlockItem * childItem = new BlockItem(&m_context,newProxyItem,this);
        childItem->setCategory(newRealModelPointer->child(i)->category());
        childItem->setBlockXPosition(newRealModelPointer->child(i)->blockXPosition());
        childItem->setBlockYPosition(newRealModelPointer->child(i)->blockYPosition());
        childItem->equation()->setEquationString(
                    newRealModelPointer->child(i)->equation()->getEquationString()
                    );
        childItem->setRealModelPointer(newRealModelPointer->child(i));
        m_proxyRoot->appendChild(childItem);
    }
}

void BlockModel::solveEquations(){
    try {
        EquationSolver equationSolver(&m_context);
        equationSolver.solveEquations(m_root);
    }  catch (...) {
        qDebug()<<"Solver Error";
    }

}

int BlockModel::maxBlockX()
{
    int blockX = 0;
    for ( int i = 0 ; i < m_proxyRoot->childCount() ; i++ ) {
        int newBlockX = m_proxyRoot->child(i)->blockXPosition();
        if(blockX<newBlockX){
            blockX = newBlockX;
        }
    }
    return blockX;
}

int BlockModel::maxBlockY()
{
    int blockY = 0;
    for ( int i = 0 ; i < m_proxyRoot->childCount() ; i++ ) {
        int newBlockX = m_proxyRoot->child(i)->blockYPosition();
        if(blockY<newBlockX){
            blockY = newBlockX;
        }
    }
    return blockY;
}
