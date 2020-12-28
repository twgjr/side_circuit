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
    return m_proxyRoot->childBlock(blockIndex);
}

Port *DataSource::proxyPort(int blockIndex, int portIndex)
{
    return m_proxyRoot->childBlock(blockIndex)->portAt(portIndex);
}

void DataSource::newProxyRoot(BlockItem *newProxyRoot)
{
//    // set old proxy children parents to nullptr (point to nothing)
//    for ( int i = 0 ; i < m_proxyRoot->proxyChildCount() ; i++ ) {
//        m_proxyRoot->proxyChild(i)->clearProxyParent();
//    }

//    // clear the proxy children pointers in old proxy root (point to nothing)
//    m_proxyRoot->clearProxyChildren();

    // set new proxy root pointer
    m_proxyRoot=newProxyRoot;

//    // append all proxy child pointers for item pointed by proxy parent
//    for ( int i = 0 ; i < newProxyRoot->childCount() ; i++ ) {
//        m_proxyRoot->appendProxyChild(newProxyRoot->child(i));

//        //also clear the grandchildren proxy pointers
//        m_proxyRoot->proxyChild(i)->clearProxyChildren();
//    }
}

void DataSource::appendBlock(int x, int y)
{
    emit beginInsertBlock(m_proxyRoot->childBlockCount());
    BlockItem *childItem = new BlockItem(&m_context,nullptr,this);
    childItem->setBlockXPosition(x);
    childItem->setBlockYPosition(y);
    m_proxyRoot->appendBlockChild(childItem);
    //m_proxyRoot->appendProxyChild(childItem);
    emit endInsertBlock();
}

void DataSource::downLevel(int blockIndex)
{
    // change the block connected to the reset signals before changing the
    // proxy root. if not, the pointer will be lost and the reset
    // cannot complete, problem is with portmodel and block item signals connect

    emit beginResetBlockModel();
    newProxyRoot(m_proxyRoot->childBlock(blockIndex));
    emit endResetBlockModel();
}

void DataSource::upLevel()
{
    if(m_proxyRoot->parentBlock()!=nullptr){
        // parent is valid, cannot go higher than actual root
        emit beginResetBlockModel();
        newProxyRoot(m_proxyRoot->parentBlock());
        emit endResetBlockModel();
    }
}

void DataSource::printFullTree(BlockItem *rootItem, int depth)
{
    //iterate through all children and load equations then recurse to children
    if(rootItem->parentBlock() == nullptr){
        qDebug() << "ROOT at level:"<<depth;
    }
    for (int i = 0 ; i < rootItem->childBlockCount() ; i++) {
        //is a leaf, then print and return, else continue to traverse the tree
        QString spacer;
        for (int j = 0 ; j < depth+1 ; j++){
            spacer+="    ";
        }
        qDebug()<<spacer<<"-> Depth - Child: "<<depth+1<<"-"<<rootItem->childBlock(i)->childBlockNumber();
        printFullTree(rootItem->childBlock(i),depth+1);
    }
}

void DataSource::printBlock(int blockIndex)
{
    qDebug()<<"ID: " << m_proxyRoot->childBlock(blockIndex)->id();
    qDebug()<<"Category: " << m_proxyRoot->childBlock(blockIndex)->description();
    qDebug()<<"Position: " << m_proxyRoot->childBlock(blockIndex)->blockXPosition()
           << " x "
           << m_proxyRoot->childBlock(blockIndex)->blockYPosition();
    qDebug()<<"Equation: " << m_proxyRoot->childBlock(blockIndex)->equationString();

}

int DataSource::distanceFromRoot() const
{
    int count = 0;

    if(m_proxyRoot->parentBlock()==nullptr){
        return count; //at the real root
    }

    BlockItem * realItem = m_proxyRoot;
    realItem = realItem->parentBlock();
    count+=1;
    while(realItem->parentBlock()!=nullptr){
        realItem = realItem->parentBlock();
        count+=1;
    }
    return count;
}

//int DataSource::numChildren(int blockIndex)
//{
//    return m_proxyRoot->childBlock(blockIndex)->childBlockCount();
//}

void DataSource::deleteBlock(int blockIndex)
{
    emit beginResetBlockModel();
    m_proxyRoot->removeBlockChild(blockIndex);
//    m_proxyRoot->removeProxyChild(blockIndex);
    emit endResetBlockModel();
}

void DataSource::addPort(int blockIndex, int side, int position)
{
    m_proxyRoot->childBlock(blockIndex)->addPort(side,position);
}

void DataSource::deletePort(int blockIndex, int portIndex)
{
    m_proxyRoot->childBlock(blockIndex)->removePort(portIndex);
}

void DataSource::startLink(int blockIndex, int portIndex)
{
    m_proxyRoot->childBlock(blockIndex)->portAt(portIndex)->startLink();
}

void DataSource::deleteLink(int blockIndex, int portIndex, int linkIndex)
{
    m_proxyRoot->childBlock(blockIndex)->portAt(portIndex)->removeLink(linkIndex);
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
    for ( int i = 0 ; i < m_proxyRoot->childBlockCount() ; i++ ) {
        int newBlockX = m_proxyRoot->childBlock(i)->blockXPosition();
        if(blockX<newBlockX){
            blockX = newBlockX;
        }
    }
    return blockX;
}

int DataSource::maxBlockY()
{
    int blockY = 0;
    for ( int i = 0 ; i < m_proxyRoot->childBlockCount() ; i++ ) {
        int newBlockX = m_proxyRoot->childBlock(i)->blockYPosition();
        if(blockY<newBlockX){
            blockY = newBlockX;
        }
    }
    return blockY;
}
