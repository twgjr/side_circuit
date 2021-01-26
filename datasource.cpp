#include "datasource.h"

DataSource::DataSource(QObject *parent) : QObject(parent),
    m_pendingConnectPort(nullptr),
    m_pendingConnectLink(nullptr)
{
    //qDebug()<<"DataSource created";
    //by default the root node is always a block
    m_root = new DiagramItem(DItemTypes::Enum::BlockItem,&m_context,nullptr,this);  // real root is empty
    m_proxyRoot = m_root; // always start at empty top level
}

DataSource::~DataSource()
{
    //qDebug()<<"DataSource destroyed";
}

DiagramItem *DataSource::proxyRoot()
{
    return m_proxyRoot;
}

DiagramItem *DataSource::proxyChild(int blockIndex)
{
    return m_proxyRoot->childItemAt(blockIndex);
}

Port *DataSource::proxyPort(int blockIndex, int portIndex)
{
    return m_proxyRoot->childItemAt(blockIndex)->portAt(portIndex);
}

void DataSource::newProxyRoot(DiagramItem *newProxyRoot)
{
    m_proxyRoot=newProxyRoot;
}

void DataSource::appendDiagramItem(int type, int x, int y)
{
    emit beginResetDiagramItems();
    m_proxyRoot->addItemChild(type,x,y);
    emit endResetDiagramItems();
}

void DataSource::deleteDiagramItem(int index)
{
    emit beginResetDiagramItems();
    m_proxyRoot->removeItemChild(index);
    emit endResetDiagramItems();
}

void DataSource::addEquation()
{
    emit beginResetEquations();
    m_proxyRoot->addEquation();
    emit endResetEquations();
}

void DataSource::deleteEquation(int index)
{
    emit beginResetEquations();
    m_proxyRoot->removeEquation(index);
    emit endResetEquations();
}

void DataSource::downLevel(int blockIndex)
{
    emit beginResetDiagramItems();
    emit beginResetEquations();
    newProxyRoot(m_proxyRoot->childItemAt(blockIndex));
    emit endResetEquations();
    emit endResetDiagramItems();
}

void DataSource::upLevel()
{
    if(m_proxyRoot->parentItem()!=nullptr){
        // parent is valid, cannot go higher than actual root
        emit beginResetDiagramItems();
        emit beginResetEquations();
        newProxyRoot(m_proxyRoot->parentItem());
        emit endResetEquations();
        emit endResetDiagramItems();
    }
}

void DataSource::printFullTree(DiagramItem *rootItem, int depth)
{
    //iterate through all children and load equations then recurse to children
    if(rootItem->parentItem() == nullptr){
        qDebug() << "ROOT at level:"<<depth;
    }
    for (int i = 0 ; i < rootItem->childItemCount() ; i++) {
        //is a leaf, then print and return, else continue to traverse the tree
        QString spacer;
        for (int j = 0 ; j < depth+1 ; j++){
            spacer+="    ";
        }
        qDebug()<<spacer<<"-> Depth - Child: "<<depth+1<<"-"<<rootItem->childItemAt(i)->childItemNumber();
        printFullTree(rootItem->childItemAt(i),depth+1);
    }
}

void DataSource::printBlock(int blockIndex)
{
    qDebug()<<"ID: " << m_proxyRoot->childItemAt(blockIndex)->id();
    qDebug()<<"Position: " << m_proxyRoot->childItemAt(blockIndex)->xPos()
           << " x "
           << m_proxyRoot->childItemAt(blockIndex)->yPos();
}

int DataSource::distanceFromRoot() const
{
    int count = 0;

    if(m_proxyRoot->parentItem()==nullptr){
        return count; //at the real root
    }

    DiagramItem * realItem = m_proxyRoot;
    realItem = realItem->parentItem();
    count+=1;
    while(realItem->parentItem()!=nullptr){
        realItem = realItem->parentItem();
        count+=1;
    }
    return count;
}

void DataSource::addPort(int index, int side, int position)
{
    m_proxyRoot->childItemAt(index)->addPort(side,position);
}

void DataSource::deletePort(int index, int portIndex)
{
    m_proxyRoot->childItemAt(index)->removePort(portIndex);
}

void DataSource::startLink(int index, int portIndex)
{
    m_proxyRoot->childItemAt(index)->portAt(portIndex)->startLink();
}

void DataSource::deleteLink(int index, int portIndex, int linkIndex)
{
    m_proxyRoot->childItemAt(index)->portAt(portIndex)->removeLink(linkIndex);
}

void DataSource::endLinkFromLink( Link* thisLink )
{
    if(m_pendingConnectPort){
        m_pendingConnectPort->appendConnectedLink(thisLink);
        thisLink->setEndPort(m_pendingConnectPort);
        //cleanup the buffer pointers when done connecting
        m_pendingConnectPort = nullptr;
        m_pendingConnectLink = nullptr;
    } else {
        m_pendingConnectLink = thisLink;
    }
}

void DataSource::endLinkFromPort( Port* thisPort )
{
    if(m_pendingConnectLink){
        thisPort->appendConnectedLink(m_pendingConnectLink);
        m_pendingConnectLink->setEndPort(thisPort);
        //cleanup the buffer pointers when done connecting
        m_pendingConnectPort = nullptr;
        m_pendingConnectLink = nullptr;
    } else {
        m_pendingConnectPort = thisPort;
    }
}

void DataSource::disconnectPortfromLink(Link *thisLink)
{
    thisLink->disconnectEndPort();
}

void DataSource::resetLinkstoPort(Port *thisPort)
{
    thisPort->resetLinkModel();
}

void DataSource::resetConnectedLinkstoPort(Port *thisPort)
{
    for(int i = 0; i < thisPort->connectedLinks().size(); i++){
        thisPort->connectedLinks()[i]->startPort()->resetLinkModel();
    }
}

void DataSource::solveEquations()
{
    try {
        emit beginResetResults();
        EquationSolver equationSolver(&m_context);
        equationSolver.solveEquations(m_root);
        emit endResetResults();
    }  catch (...) {
        qDebug()<<"Solver Error";
    }
}

int DataSource::maxItemX()
{
    int blockX = 0;
    for ( int i = 0 ; i < m_proxyRoot->childItemCount() ; i++ ) {
        int newBlockX = m_proxyRoot->childItemAt(i)->xPos();
        if(blockX<newBlockX){
            blockX = newBlockX;
        }
    }
    return blockX;
}

int DataSource::maxItemY()
{
    int blockY = 0;
    for ( int i = 0 ; i < m_proxyRoot->childItemCount() ; i++ ) {
        int newBlockX = m_proxyRoot->childItemAt(i)->yPos();
        if(blockY<newBlockX){
            blockY = newBlockX;
        }
    }
    return blockY;
}
