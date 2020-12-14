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
    qDebug()<<"Block Item created.";
}

BlockItem::~BlockItem()
{
    qDebug()<<"Block Item destroyed.";
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

//bool BlockItem::insertChildren(int position, int count, int columns)
//{
//    if (position < 0 || position > m_children.size())
//        return false;

//    for (int row = 0; row < count; ++row) {
//        QVector<QVariant> data(columns);
//        BlockItem *item = new BlockItem(m_context,this);
//        m_children.insert(position, item);
//    }
//    return true;
//}

//bool BlockItem::removeChildren(int position, int count)
//{
//    if (position < 0 || position + count > m_children.size())
//        return false;

//    for (int row = 0; row < count; ++row)
//        delete m_children.takeAt(position);

//    return true;
//}

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

void BlockItem::setProxyParent(BlockItem *proxyParent) {m_proxyParent = proxyParent;}

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
QVector<BlockItem *> BlockItem::proxyChildren() {return m_proxyChildren;}

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
    delete m_proxyChildren[modelIndex];
    m_proxyChildren.remove(modelIndex);
}

QVector<Port *> BlockItem::ports() const {return m_ports;}
void BlockItem::addPort(int side, int position){
    Port * newPort = new Port(this);
    BlockItem * thisItem = static_cast<BlockItem*>(this);
    newPort->setBlockParent(thisItem);
    newPort->setSide(side);
    newPort->setPosition(position);
    m_ports.append(newPort);
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
