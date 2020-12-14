#include "blockmodel.h"

BlockModel::BlockModel(QObject *parent) : QAbstractItemModel(parent)
{
    //qDebug()<<"BlockModel object created.";

    m_root = new BlockItem(&m_context,nullptr,this);  // real root is empty
    m_proxyRoot = m_root;//new BlockItem(&m_context,nullptr,this);
    //newProxyRoot(m_root);  //clone real root into proxy root

    //set the basic roles for access of item properties in QML
    m_roles[BlockItem::DescriptionDataRole]="description";
    m_roles[BlockItem::IDDataRole]="id";
    m_roles[BlockItem::BlockXPositionRole]="blockXPosition";
    m_roles[BlockItem::BlockYPositionRole]="blockYPosition";
    m_roles[BlockItem::blockHeightRole]="blockHeight";
    m_roles[BlockItem::blockWidthRole]="blockWidth";
    m_roles[BlockItem::EquationRole]="equationString";
}

BlockModel::~BlockModel()
{
    qDebug()<<"Block Model destroyed.";
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
        case BlockItem::DescriptionDataRole:
            if(blockItem->description() != value.toString()){
                blockItem->setDescription(value.toString());
            }
            break;
        case BlockItem::IDDataRole: break;
        case BlockItem::BlockXPositionRole:
            if(blockItem->blockXPosition() != value.toInt()){
                blockItem->setBlockXPosition(value.toInt());
            }
            break;
        case BlockItem::BlockYPositionRole:
            if(blockItem->blockYPosition() != value.toInt()){
                blockItem->setBlockYPosition(value.toInt());
            }
            break;
        case BlockItem::EquationRole:
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
    return m_proxyRoot;
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
    if (proxyParentItem == m_proxyRoot){
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

void BlockModel::appendBlock(int x, int y)
{
    int pos = m_proxyRoot->childCount();
    QModelIndex proxyParentIndex = qIndexOfBlock(m_proxyRoot);
    beginInsertRows(proxyParentIndex, pos, pos);

    BlockItem *childItem = new BlockItem(&m_context,nullptr,this);
    childItem->setBlockXPosition(x);
    childItem->setBlockYPosition(y);
    m_proxyRoot->appendChild(childItem);
    m_proxyRoot->appendProxyChild(childItem);
    endInsertRows();
    //printProxyTree(m_proxyRoot,0);
    //printFullTree(m_root,0);
}

void BlockModel::downLevel(int modelIndex)
{
    beginResetModel();
    newProxyRoot(m_proxyRoot->child(modelIndex));
    endResetModel();
    //printProxyTree(m_proxyRoot,0);
    //printFullTree(m_root,0);
}

void BlockModel::upLevel()
{
    if(m_proxyRoot->parentItem()!=nullptr){
        // parent is valid, cannot go higher than actual root
        beginResetModel();
        newProxyRoot(m_proxyRoot->parentItem());
        endResetModel();
    }
    //printProxyTree(m_proxyRoot,0);
    //printFullTree(m_root,0);
}

void BlockModel::printProxyTree(BlockItem *rootItem, int depth)
{
    //iterate through all children and load equations then recurse to children
    if(rootItem->proxyParent() == nullptr){
        qDebug() << "PROXY ROOT at level:"<<depth;
    }
    for (int i = 0 ; i < rootItem->proxyChildCount() ; i++) {
        //is a leaf, then print and return, else continue to traverse the tree
        QString spacer;
        for (int j = 0 ; j < depth+1 ; j++){
            spacer+="    ";
        }
        qDebug()<<spacer<<"-> Depth - Child: "<<depth+1<<"-"<<rootItem->proxyChild(i)->childNumber();
        printProxyTree(rootItem->proxyChild(i),depth+1);
    }
}

void BlockModel::printFullTree(BlockItem * rootItem, int depth)
{
    //iterate through all children and load equations then recurse to children
    if(rootItem->parentItem() == nullptr){
        qDebug() << "ROOT at level:"<<depth;
    }
    for (int i = 0 ; i < rootItem->childCount() ; i++) {
        //is a leaf, then print and return, else continue to traverse the tree
        QString spacer;
        for (int j = 0 ; j < depth+1 ; j++){
            spacer+="    ";
        }
        QString hasProxy;
        if(rootItem->proxyChildCount()>0){
            hasProxy = "*";
        }
        qDebug()<<spacer<<"-> Depth - Child: "<<depth+1<<"-"<<rootItem->child(i)->childNumber()<<hasProxy;
        printFullTree(rootItem->child(i),depth+1);
    }
}

void BlockModel::printBlock(int modelIndex)
{
    qDebug()<<"ID: " << m_proxyRoot->child(modelIndex)->id();
    qDebug()<<"Category: " << m_proxyRoot->child(modelIndex)->description();
    qDebug()<<"Position: " << m_proxyRoot->child(modelIndex)->blockXPosition()
           << " x "
           << m_proxyRoot->child(modelIndex)->blockYPosition();
    qDebug()<<"Equation: " << m_proxyRoot->child(modelIndex)->equationString();
}

int BlockModel::distanceFromRoot() const
{
    int count = 0;

    if(m_proxyRoot->parentItem()==nullptr){
        qDebug()<<"root";
        return count; //at the real root
    }

    BlockItem * realItem = m_proxyRoot;
    realItem = realItem->parentItem();
    count+=1;
    while(realItem->parentItem()!=nullptr){
        realItem = realItem->parentItem();
        count+=1;
    }
    qDebug()<<"level is: "<<count;

    return count;

}

int BlockModel::numChildren(int modelIndex)
{
    return m_proxyRoot->child(modelIndex)->childCount();
}

void BlockModel::deleteBlock(int modelIndex)
{
    beginResetModel();
    m_proxyRoot->removeChild(modelIndex);
    m_proxyRoot->removeProxyChild(modelIndex);
    endResetModel();
}

void BlockModel::addPort(int modelIndex, int side, int position)
{
    beginResetModel();
    m_proxyRoot->child(modelIndex)->addPort(side,position);
    endResetModel();
    qDebug()<<"Added port on side:"<<side<<"at"<<position;
}

void BlockModel::newProxyRoot(BlockItem *newProxyRoot)
{
    // set old proxy children parents to nullptr (point to nothing)
    for ( int i = 0 ; i < m_proxyRoot->proxyChildCount() ; i++ ) {
        m_proxyRoot->proxyChild(i)->clearProxyParent();
    }

    // clear the proxy children pointers in old proxy root (point to nothing)
    m_proxyRoot->clearProxyChildren();


    // set new proxy root pointer
    m_proxyRoot=newProxyRoot;
    //m_proxyRoot->setProxyParent(nullptr);

    // append all proxy child pointers for item pointed by proxy parent
    for ( int i = 0 ; i < newProxyRoot->childCount() ; i++ ) {
        m_proxyRoot->appendProxyChild(newProxyRoot->child(i));

        //also clear the grandchildren proxy pointers
        m_proxyRoot->proxyChild(i)->clearProxyChildren();
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
