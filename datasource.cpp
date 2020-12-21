#include "datasource.h"

DataSource::DataSource(QObject *parent) : QObject(parent)
{
    //qDebug()<<"DataSource created";
    m_root = new BlockItem(&m_context,nullptr,this);  // real root is empty
    m_proxyRoot = m_root;

    //appendBlock(50,50);
    //appendBlock(200,200);
    //addPort(0,0,50);
    //addPort(1,3,50);
}

DataSource::~DataSource()
{
    //qDebug()<<"DataSource destroyed";
}

BlockItem *DataSource::proxyRoot(){
    return m_proxyRoot;
}

BlockItem *DataSource::proxyChild(int blockIndex)
{
    return m_proxyRoot->child(blockIndex);
}

void DataSource::newProxyRoot(BlockItem *newProxyRoot)
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

void DataSource::appendBlock(int x, int y)
{

    beginInsertBlock(m_proxyRoot->childCount());
    BlockItem *childItem = new BlockItem(&m_context,nullptr,this);
    childItem->setBlockXPosition(x);
    childItem->setBlockYPosition(y);
    m_proxyRoot->appendChild(childItem);
    m_proxyRoot->appendProxyChild(childItem);
    endInsertBlock();
}

void DataSource::downLevel(int blockIndex)
{
    beginResetBlockModel();
    newProxyRoot(m_proxyRoot->child(blockIndex));
    endResetBlockModel();
}

void DataSource::upLevel()
{
    if(m_proxyRoot->parentItem()!=nullptr){
        // parent is valid, cannot go higher than actual root
        beginResetBlockModel();
        newProxyRoot(m_proxyRoot->parentItem());
        endResetBlockModel();
    }
}

void DataSource::printProxyTree(BlockItem *rootItem, int depth)
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

void DataSource::printFullTree(BlockItem *rootItem, int depth)
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

void DataSource::printBlock(int blockIndex)
{
    qDebug()<<"ID: " << m_proxyRoot->child(blockIndex)->id();
    qDebug()<<"Category: " << m_proxyRoot->child(blockIndex)->description();
    qDebug()<<"Position: " << m_proxyRoot->child(blockIndex)->blockXPosition()
           << " x "
           << m_proxyRoot->child(blockIndex)->blockYPosition();
    qDebug()<<"Equation: " << m_proxyRoot->child(blockIndex)->equationString();

}

int DataSource::distanceFromRoot() const
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

int DataSource::numChildren(int blockIndex)
{
    return m_proxyRoot->child(blockIndex)->childCount();
}

void DataSource::deleteBlock(int blockIndex)
{
    beginResetBlockModel();
    m_proxyRoot->removeChild(blockIndex);
    m_proxyRoot->removeProxyChild(blockIndex);
    endResetBlockModel();
}

void DataSource::addPort(int blockIndex, int side, int position)
{
    //delete row then insert to reset only the affect block
    beginRemoveBlock(blockIndex);
    // don't actually remove the block
    endRemoveBlock();
    beginInsertBlock(blockIndex);
    m_proxyRoot->child(blockIndex)->addPort(side,position);
    endInsertBlock();
}

void DataSource::solveEquations()
{
    try {
        EquationSolver equationSolver(&m_context);
        equationSolver.solveEquations(m_root);
    }  catch (...) {
        qDebug()<<"Solver Error";
    }
}

int DataSource::maxBlockX()
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

int DataSource::maxBlockY()
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
