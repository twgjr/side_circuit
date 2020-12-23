#include "blockitem.h"



BlockItem::BlockItem(z3::context *context,
                     BlockItem *parent,
                     QObject * qobjparent) :
    QObject(qobjparent),
    m_parentItem(parent),
    m_proxyParent(nullptr),
    m_context(context),
    m_blockType(Block),
    m_description(""),
    m_blockXPosition(0),
    m_blockYPosition(0),
    m_equation(context)
{
    qDebug()<<"Created: "<<this<<" with Qparent: "<<qobjparent;
}

BlockItem::~BlockItem()
{
    qDebug()<<"Deleted: "<<this;
}

BlockItem *BlockItem::parentItem() {return m_parentItem;}
void BlockItem::setParentItem(BlockItem *parentItem) {m_parentItem = parentItem;}

BlockItem *BlockItem::child(int index)
{
    if(index < 0 || index >= m_children.length())
        return nullptr;
    return m_children.at(index);
}

int BlockItem::childNumber() const
{
    if (m_parentItem)
        return m_parentItem->m_children.indexOf(const_cast<BlockItem*>(this));
    return 0;
}

int BlockItem::childCount() const {return m_children.count();}

bool BlockItem::appendChild(BlockItem *item)
{
    item->m_parentItem = this;
    item->setParent(this);  //to automatically delete if QObject parent destroyed
    m_children.append(item);
    return true;
}

void BlockItem::removeChild(int modelIndex)
{
    delete m_children[modelIndex];
    m_children.remove(modelIndex);
}

int BlockItem::columnCount() const {return 1;} // columns not used

void BlockItem::clearProxyParent()
{
    m_proxyParent = nullptr;
}

BlockItem *BlockItem::proxyChild(int index)
{
    if(index < 0 || index >= m_proxyChildren.length())
        return nullptr;
    return m_proxyChildren.at(index);
}

int BlockItem::proxyChildNumber() const
{
    if (m_proxyParent)
        return m_proxyParent->m_proxyChildren.indexOf(const_cast<BlockItem*>(this));
    return 0;
}

int BlockItem::proxyChildCount() const
{
    return m_proxyChildren.count();
}
BlockItem *BlockItem::proxyParent() {return m_proxyParent;}

void BlockItem::clearProxyChildren()
{
    while ( !m_proxyChildren.isEmpty() ) {
        m_proxyChildren.removeLast();
    }
}

void BlockItem::appendProxyChild(BlockItem *item)
{
    item->m_proxyParent = this;
    item->setParent(this);  //to automatically delete if QObject parent destroyed
    m_proxyChildren.append(item);
}

void BlockItem::removeProxyChild(int modelIndex)
{
    m_proxyChildren.remove(modelIndex);
}

QVector<Port *> BlockItem::ports() {return m_ports;}

Port *BlockItem::portAt(int portIndex)
{
    return m_ports[portIndex];
}
void BlockItem::addPort(int side, int position){
    qDebug()<<"next port Index to be added "<< m_ports.count();
    Port * newPort = new Port(this);
    BlockItem * thisItem = static_cast<BlockItem*>(this);
    newPort->setBlockParent(thisItem);
    newPort->setSide(side);
    newPort->setPosition(position);
    emit beginInsertPort(m_ports.count());
    m_ports.append(newPort);
    emit endInsertPort();
}

void BlockItem::removePort(int portIndex)
{
    qDebug()<<"port Index to be removed "<< portIndex;
    emit beginRemovePort(portIndex);
    delete m_ports[portIndex];
    m_ports.remove(portIndex);
    emit endRemovePort();
}

int BlockItem::portCount()
{
    return m_ports.count();
}

void BlockItem::setContext(z3::context *context) {m_context = context;}
z3::context *BlockItem::context() const {return m_context;}

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
void BlockItem::setBlockType(int blockType) {m_blockType = blockType;}
int BlockItem::blockType() const {return m_blockType;}
QString BlockItem::description() const {return m_description;}
void BlockItem::setDescription(QString category) {m_description = category;}
int BlockItem::id() const {return childNumber();}
int BlockItem::blockXPosition() const {return m_blockXPosition;}
void BlockItem::setBlockXPosition(int blockXPosition){m_blockXPosition = blockXPosition;}
int BlockItem::blockYPosition() const {return m_blockYPosition;}
void BlockItem::setBlockYPosition(int blockYPosition){m_blockYPosition = blockYPosition;}
Equation * BlockItem::equation(){return &m_equation;}
QString BlockItem::equationString() {return m_equation.getEquationString();}
void BlockItem::setEquationString(QString equationString)
{
    if (m_equation.getEquationString() == equationString)
        return;
    m_equation.setEquationString(equationString);
}
int BlockItem::blockWidth() const {return m_blockWidth;}
int BlockItem::blockHeight() const {return m_blockHeight;}
void BlockItem::setBlockWidth(int blockWidth) {m_blockWidth = blockWidth;}
void BlockItem::setblockHeight(int blockHeight) {m_blockHeight = blockHeight;}
