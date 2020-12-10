#include "blockitem.h"



BlockItem::BlockItem(z3::context *context,
                     BlockItem *parent,
                     QObject * qobjparent) :
    QObject(qobjparent),
    m_category(""),
    m_blockXPosition(0),
    m_blockYPosition(0),
    m_equation(context),
    m_context(context),
    m_parent(parent),
    m_realModelPointer(nullptr)
{
    qDebug()<<"Block Item created.";
}

BlockItem::~BlockItem()
{
    qDeleteAll(m_children);
    m_children.clear();
    qDebug()<<"Block Item destroyed.";
}

BlockItem *BlockItem::parentItem()
{
    return m_parent;
}

BlockItem *BlockItem::child(int index)
{
    if(index < 0 || index >= m_children.length())
        return nullptr;
    return m_children.at(index);
}

int BlockItem::childNumber() const
{
    if (m_parent)
        return m_parent->m_children.indexOf(const_cast<BlockItem*>(this));
    return 0;
}

int BlockItem::childCount() const
{
    return m_children.count();
}

bool BlockItem::insertChildren(int position, int count, int columns)
{
    if (position < 0 || position > m_children.size())
        return false;

    for (int row = 0; row < count; ++row) {
        QVector<QVariant> data(columns);
        BlockItem *item = new BlockItem(m_context,this);
        m_children.insert(position, item);
    }
    return true;
}

bool BlockItem::removeChildren(int position, int count)
{
    if (position < 0 || position + count > m_children.size())
        return false;

    for (int row = 0; row < count; ++row)
        delete m_children.takeAt(position);

    return true;
}

int BlockItem::columnCount() const
{
    return 0; // 0 column is default size nothing stored as QVariantList data needing columns
}

bool BlockItem::appendChild(BlockItem *item)
{
    item->m_parent = this;
    item->setParent(this);  //to automatically delete if QObject parent destroyed
    m_children.append(item);
    return true;
}

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

QString BlockItem::category() const {return m_category;}
void BlockItem::setCategory(QString category) {
    m_category = category;
}

int BlockItem::id() const {return childNumber();}

int BlockItem::blockXPosition() const {return m_blockXPosition;}
void BlockItem::setBlockXPosition(int blockXPosition){
    m_blockXPosition = blockXPosition;
}

int BlockItem::blockYPosition() const {return m_blockYPosition;}
void BlockItem::setBlockYPosition(int blockYPosition){
    m_blockYPosition = blockYPosition;
}

Equation * BlockItem::equation(){return &m_equation;}

QString BlockItem::equationString() {return m_equation.getEquationString();}
void BlockItem::setEquationString(QString equationString)
{
    if (m_equation.getEquationString() == equationString)
        return;
    m_equation.setEquationString(equationString);
}

BlockItem *BlockItem::realModelPointer()
{
    return m_realModelPointer;
}

void BlockItem::setRealModelPointer(BlockItem *realModelPointer)
{
    m_realModelPointer = realModelPointer;
}
