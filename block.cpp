#include "block.h"

Block::Block(z3::context *context,
                     Block *parentBlock,
                     QObject * parent) :
    QObject(parent),
    m_parentItem(parentBlock),
    m_context(context),
    m_type("block"),
    m_description(""),
    m_xPos(0),
    m_yPos(0),
    m_itemWidth(0),
    m_itemHeight(0)
{
    //qDebug()<<"Created: "<<this<<" with Qparent: "<<qobjparent;
}

Block::~Block()
{
    //qDebug()<<"Deleted: "<<this;
}

Block *Block::parentBlock() {return m_parentItem;}
void Block::setParentBlock(Block *parentBlock) {m_parentItem = parentBlock;}

Block *Block::childBlockAt(int index)
{
    if(index < 0 || index >= m_blockChildren.size()){
        return nullptr;
    }
    return m_blockChildren.at(index);
}

int Block::childBlockNumber() const
{
    if (m_parentItem)
        return m_parentItem->m_blockChildren.indexOf(const_cast<Block*>(this));
    return 0;
}

int Block::childBlockCount() const {return m_blockChildren.count();}

void Block::addBlockChild(int x, int y)
{
    Block *childItem = new Block(m_context,this,this);
    childItem->setXPos(x);
    childItem->setYPos(y);
    m_blockChildren.append(childItem);
}

void Block::removeBlockChild(int modelIndex)
{
    delete m_blockChildren[modelIndex];
    m_blockChildren.remove(modelIndex);
}

Port *Block::portAt(int portIndex)
{
    return m_ports[portIndex];
}
void Block::addPort(int side, int position){
    Port * newPort = new Port(this);
    Block * thisItem = static_cast<Block*>(this);
    newPort->setBlockParent(thisItem);
    newPort->setSide(side);
    newPort->setPosition(position);
    emit beginInsertPort(m_ports.count());
    m_ports.append(newPort);
    emit endInsertPort();
}

void Block::removePort(int portIndex)
{
    qDebug()<<"port Index to be removed "<< portIndex;
    emit beginRemovePort(portIndex);
    delete m_ports[portIndex];
    m_ports.remove(portIndex);
    emit endRemovePort();
}

int Block::portCount()
{
    return m_ports.count();
}


QVector<Equation*> Block::equations()
{
    return m_equationChildren;
}

int Block::equationCount()
{
    return m_equationChildren.count();
}

Equation *Block::childEquationAt(int index)
{
    return m_equationChildren[index];
}

void Block::addEquation(int x, int y)
{
    qDebug()<<"create equation object";
    Equation * newEquation = new Equation(m_context,this);
    newEquation->setXPos(x);
    newEquation->setYPos(y);
    m_equationChildren.append(newEquation);
}

void Block::removeEquation(int index)
{
    delete m_equationChildren[index];
    m_equationChildren.remove(index);
}

void Block::setContext(z3::context *context) {m_context = context;}
z3::context *Block::context() const {return m_context;}

//void BlockItem::jsonRead(QJsonObject &json)
//{
//    if (json.contains("category") && json["category"].isString()){
//        m_category = json["category"].toString();
//    }else{
//        qDebug()<<"Could not load category";
//    }

//    //    if (json.contains("Block_ID") && json["Block_ID"].isDouble()){
//    //        m_id = json["Block_ID"].toInt();
//    //    }else{
//    //        qDebug()<<"Could not load id";
//    //    }

//    if (json.contains("BlockXPosition") && json["BlockXPosition"].isDouble()){
//        m_blockXPosition = json["BlockXPosition"].toInt();
//    }else{
//        qDebug()<<"Could not load BlockXPosition";
//    }

//    if (json.contains("BlockYPosition") && json["BlockYPosition"].isDouble()){
//        m_blockYPosition = json["BlockYPosition"].toInt();
//    }else{
//        qDebug()<<"Could not load BlockYPosition";
//    }
//    if (json.contains("Equation") && json["Equation"].isString()){
//        m_equation.setEquationString(json["Equation"].toString());
//    }else{
//        qDebug()<<"Could not load Equation";
//    }
//}

//void BlockItem::jsonWrite(QJsonObject &json)
//{
//    //json["Block_ID"] = m_id;
//    json["category"] = m_category;
//    json["BlockXPosition"] = m_blockXPosition;
//    json["BlockYPosition"] = m_blockYPosition;
//    json["Equation"] = m_equation.getEquationString();
//    //Commented code sample for how to add an array
//    /*
//    QJsonArray npcArray;
//    for (const Character &npc : mNpcs) {
//        QJsonObject npcObject;
//        npc.write(npcObject);
//        npcArray.append(npcObject);
//    }
//    json["npcs"] = npcArray;
//    */
//}

QString Block::description() const {return m_description;}
void Block::setDescription(QString category) {m_description = category;}
int Block::id() const {return childBlockNumber();}
int Block::xPos() const {return m_xPos;}
void Block::setXPos(int blockXPosition){m_xPos = blockXPosition;}
int Block::yPos() const {return m_yPos;}
void Block::setYPos(int blockYPosition){m_yPos = blockYPosition;}
int Block::itemWidth() const {return m_itemWidth;}
int Block::itemHeight() const {return m_itemHeight;}
void Block::setItemWidth(int blockWidth) {m_itemWidth = blockWidth;}
void Block::setblockHeight(int blockHeight) {m_itemHeight = blockHeight;}

Block *Block::proxyRoot()
{
    return m_proxyRoot;
}

void Block::setProxyRoot(Block *proxyRoot)
{
    if (m_proxyRoot == proxyRoot)
        return;

    m_proxyRoot = proxyRoot;
    emit proxyRootChanged(m_proxyRoot);
}

Block *Block::thisItem()
{
    return this;
}

void Block::setThisItem(Block *thisBlock)
{
    if (m_thisItem == thisBlock)
        return;

    m_thisItem = thisBlock;
    emit thisItemChanged(m_thisItem);
}

QString Block::type() const
{
    return m_type;
}

void Block::setType(QString type)
{
    if (m_type == type)
        return;

    m_type = type;
    emit typeChanged(m_type);
}


