#include "diagramdatasource.h"

DiagramDataSource::DiagramDataSource(QObject *parent) : QObject(parent)
{
    qDebug()<<"DiagramDataSource created";
    m_root = new BlockItem(&m_context,nullptr,this);  // real root is empty
    m_proxyRoot = m_root;

    appendBlock(50,50);
    appendBlock(200,200);
    addPort(0,0,50);
    addPort(1,3,50);
}

DiagramDataSource::~DiagramDataSource()
{
    qDebug()<<"DiagramDataSource destroyed";
}

BlockItem *DiagramDataSource::proxyRoot(){
    return m_proxyRoot;
}

BlockItem *DiagramDataSource::blockDataSource(int index)
{
    return m_proxyRoot->child(index);
}

void DiagramDataSource::newProxyRoot(BlockItem *newProxyRoot)
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

void DiagramDataSource::appendBlock(int x, int y)
{
    //int blockIndex = m_proxyRoot->childCount();
    //QModelIndex proxyParentIndex = qIndexOfBlock(m_proxyRoot);
    //beginInsertRows(proxyParentIndex, pos, pos);
    beginInsertBlock();

    BlockItem *childItem = new BlockItem(&m_context,nullptr,this);
    childItem->setBlockXPosition(x);
    childItem->setBlockYPosition(y);
    m_proxyRoot->appendChild(childItem);
    m_proxyRoot->appendProxyChild(childItem);
//    endInsertRows();
    endRemoveBlock();
    //printProxyTree(m_proxyRoot,0);
    //printFullTree(m_root,0);
}

BlockItem *DiagramDataSource::thisBlock(int blockIndex)
{
    return m_proxyRoot->child(blockIndex);

}

void DiagramDataSource::downLevel(int blockIndex)
{
    //beginResetModel();
    beginResetBlockModel();
    newProxyRoot(m_proxyRoot->child(blockIndex));
    //endResetModel();
    endResetBlockModel();
    //printProxyTree(m_proxyRoot,0);
    //printFullTree(m_root,0);
}

void DiagramDataSource::upLevel()
{
    if(m_proxyRoot->parentItem()!=nullptr){
        // parent is valid, cannot go higher than actual root
        //beginResetModel();
        beginResetBlockModel();
        newProxyRoot(m_proxyRoot->parentItem());
        //endResetModel();
        endResetBlockModel();
    }
    //printProxyTree(m_proxyRoot,0);
    //printFullTree(m_root,0);
}

void DiagramDataSource::printProxyTree(BlockItem *rootItem, int depth)
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

void DiagramDataSource::printFullTree(BlockItem *rootItem, int depth)
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

void DiagramDataSource::printBlock(int blockIndex)
{
    qDebug()<<"ID: " << m_proxyRoot->child(blockIndex)->id();
    qDebug()<<"Category: " << m_proxyRoot->child(blockIndex)->description();
    qDebug()<<"Position: " << m_proxyRoot->child(blockIndex)->blockXPosition()
           << " x "
           << m_proxyRoot->child(blockIndex)->blockYPosition();
    qDebug()<<"Equation: " << m_proxyRoot->child(blockIndex)->equationString();

}

int DiagramDataSource::distanceFromRoot() const
{
    int count = 0;

    if(m_proxyRoot->parentItem()==nullptr){
        return count; //at the real root
    }

    BlockItem * realItem = m_proxyRoot;
    realItem = realItem->parentItem();
    count+=1;
    while(realItem->parentItem()!=nullptr){
        realItem = realItem->parentItem();
        count+=1;
    }
    return count;
}

int DiagramDataSource::numChildren(int blockIndex)
{
    return m_proxyRoot->child(blockIndex)->childCount();
}

void DiagramDataSource::deleteBlock(int blockIndex)
{
    //beginResetModel();
    beginResetBlockModel();
    m_proxyRoot->removeChild(blockIndex);
    m_proxyRoot->removeProxyChild(blockIndex);
    //endResetModel();
    endResetBlockModel();
}

void DiagramDataSource::addPort(int blockIndex, int side, int position)
{
    //delete row then insert to reset only the affect block
    //QModelIndex proxyParentIndex = qIndexOfBlock(m_proxyRoot);
    //beginRemoveRows(proxyParentIndex,modelIndex,modelIndex);
    beginRemoveBlock(blockIndex);
    // don't actually remove the block
    //endRemoveRows();
    endRemoveBlock();
    //beginInsertRows(proxyParentIndex, modelIndex, modelIndex);
    beginInsertBlock();
    m_proxyRoot->child(blockIndex)->addPort(side,position);
    //endInsertRows();
    endRemoveBlock();
}

int DiagramDataSource::portCount(int blockIndex)
{
    return m_proxyRoot->proxyChild(blockIndex)->portCount();
}

int DiagramDataSource::portSide(int blockIndex, int portNum)
{
    return m_proxyRoot->proxyChild(blockIndex)->portSide(portNum);
}

int DiagramDataSource::portPosition(int blockIndex, int portNum)
{
    return m_proxyRoot->proxyChild(blockIndex)->portPosition(portNum);
}

void DiagramDataSource::solveEquations()
{
    try {
        EquationSolver equationSolver(&m_context);
        equationSolver.solveEquations(m_root);
    }  catch (...) {
        qDebug()<<"Solver Error";
    }
}

int DiagramDataSource::maxBlockX()
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

int DiagramDataSource::maxBlockY()
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
