#include "datasource.h"

DataSource::DataSource(QObject *parent) : QObject(parent)
{
    //qDebug()<<"DataSource created";
    m_root = new Block(&m_context,nullptr,this);  // real root is empty
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

Block *DataSource::proxyRoot(){
    return m_proxyRoot;
}

Block *DataSource::proxyChild(int blockIndex)
{
    return m_proxyRoot->childBlockAt(blockIndex);
}

Port *DataSource::proxyPort(int blockIndex, int portIndex)
{
    return m_proxyRoot->childBlockAt(blockIndex)->portAt(portIndex);
}

void DataSource::newProxyRoot(Block *newProxyRoot)
{
    m_proxyRoot=newProxyRoot;
}

void DataSource::appendBlock(int x, int y)
{
    emit beginResetDiagram();
    m_proxyRoot->addBlockChild(x,y);
    emit endResetDiagram();
}

void DataSource::deleteBlock(int blockIndex)
{
    emit beginResetDiagram();
    m_proxyRoot->removeBlockChild(blockIndex);
    emit endResetDiagram();
}

void DataSource::addEquation(int x, int y)
{
    qDebug()<<"begin reset equations";
    emit beginResetDiagram();
    m_proxyRoot->addEquation(x,y);
    emit endResetDiagram();
    qDebug()<<"end reset equations";
}

void DataSource::deleteEquation(int index)
{
    emit beginResetDiagram();
    m_proxyRoot->removeEquation(index);
    emit endResetDiagram();
}

void DataSource::downLevel(int blockIndex)
{
    emit beginResetDiagram();
    newProxyRoot(m_proxyRoot->childBlockAt(blockIndex));
    emit endResetDiagram();
}

void DataSource::upLevel()
{
    if(m_proxyRoot->parentBlock()!=nullptr){
        // parent is valid, cannot go higher than actual root
        emit beginResetDiagram();
        newProxyRoot(m_proxyRoot->parentBlock());
        emit endResetDiagram();
    }
}

void DataSource::printFullTree(Block *rootItem, int depth)
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
        qDebug()<<spacer<<"-> Depth - Child: "<<depth+1<<"-"<<rootItem->childBlockAt(i)->childBlockNumber();
        printFullTree(rootItem->childBlockAt(i),depth+1);
    }
}

void DataSource::printBlock(int blockIndex)
{
    qDebug()<<"ID: " << m_proxyRoot->childBlockAt(blockIndex)->id();
    qDebug()<<"Category: " << m_proxyRoot->childBlockAt(blockIndex)->description();
    qDebug()<<"Position: " << m_proxyRoot->childBlockAt(blockIndex)->xPos()
           << " x "
           << m_proxyRoot->childBlockAt(blockIndex)->yPos();
    //qDebug()<<"Equation: " << m_proxyRoot->childBlock(blockIndex)->equationString();

}

int DataSource::distanceFromRoot() const
{
    int count = 0;

    if(m_proxyRoot->parentBlock()==nullptr){
        return count; //at the real root
    }

    Block * realItem = m_proxyRoot;
    realItem = realItem->parentBlock();
    count+=1;
    while(realItem->parentBlock()!=nullptr){
        realItem = realItem->parentBlock();
        count+=1;
    }
    return count;
}



void DataSource::addPort(int blockIndex, int side, int position)
{
    m_proxyRoot->childBlockAt(blockIndex)->addPort(side,position);
}

void DataSource::deletePort(int blockIndex, int portIndex)
{
    m_proxyRoot->childBlockAt(blockIndex)->removePort(portIndex);
}

void DataSource::startLink(int blockIndex, int portIndex)
{
    m_proxyRoot->childBlockAt(blockIndex)->portAt(portIndex)->startLink();
}

void DataSource::deleteLink(int blockIndex, int portIndex, int linkIndex)
{
    m_proxyRoot->childBlockAt(blockIndex)->portAt(portIndex)->removeLink(linkIndex);
}

void DataSource::endLink(int blockIndex, int portIndex, Link* endLink)
{
    m_proxyRoot->childBlockAt(blockIndex)->portAt(portIndex)->setConnectedLink(endLink);
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
        int newBlockX = m_proxyRoot->childBlockAt(i)->xPos();
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
        int newBlockX = m_proxyRoot->childBlockAt(i)->yPos();
        if(blockY<newBlockX){
            blockY = newBlockX;
        }
    }
    return blockY;
}
