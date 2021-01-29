#include "diagramitem.h"

DiagramItem::DiagramItem(int type, z3::context *context,
             DiagramItem *parentItem,
             QObject * parent) :
    QObject(parent),
    m_parentItem(parentItem),
    m_context(context),
    m_xPos(0),
    m_yPos(0),
    m_type(type),
    m_rotation(0)
{
    //qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;
}

DiagramItem::~DiagramItem()
{
    //qDebug()<<"Deleted: "<<this;
}

DiagramItem *DiagramItem::parentItem() {return m_parentItem;}
void DiagramItem::setParentItem(DiagramItem *parentBlock) {m_parentItem = parentBlock;}

DiagramItem *DiagramItem::childItemAt(int index)
{
    if(index < 0 || index >= m_itemChildren.size()){
        return nullptr;
    }
    return m_itemChildren.at(index);
}

int DiagramItem::IndexOfItem(DiagramItem *childBlock)
{
    return m_itemChildren.indexOf(childBlock);
}

int DiagramItem::childItemNumber() const
{
    if (m_parentItem)
        return m_parentItem->m_itemChildren.indexOf(const_cast<DiagramItem*>(this));
    return 0;
}

int DiagramItem::childItemCount() const {return m_itemChildren.count();}

void DiagramItem::addItemChild(int type, int x, int y)
{
    DiagramItem *childItem = new DiagramItem(type,m_context,this,this);
    childItem->setXPos(x);
    childItem->setYPos(y);
    m_itemChildren.append(childItem);
}

void DiagramItem::removeItemChild(int modelIndex)
{
    delete m_itemChildren[modelIndex];
    m_itemChildren.remove(modelIndex);
}

Port *DiagramItem::portAt(int portIndex)
{
    return m_ports[portIndex];
}

void DiagramItem::addPort(QPointF center)
{
    Port * newPort = new Port(this);
    newPort->setItemParent(this);
    newPort->setAbsPoint(center);
    qDebug()<<newPort->absPoint();
    emit beginInsertPort(m_ports.count());
    m_ports.append(newPort);
    emit endInsertPort();
}

void DiagramItem::removePort(int portIndex)
{
    emit beginRemovePort(portIndex);
    delete m_ports[portIndex];
    m_ports.remove(portIndex);
    emit endRemovePort();
}

int DiagramItem::portCount()
{
    return m_ports.count();
}

int DiagramItem::equationCount()
{
    return m_equationChildren.count();
}

Equation *DiagramItem::equationAt(int index)
{
    return m_equationChildren[index];
}

void DiagramItem::addEquation()
{
    Equation * newEquation = new Equation(m_context,this);
    m_equationChildren.append(newEquation);
}

void DiagramItem::removeEquation(int index)
{
    delete m_equationChildren[index];
    m_equationChildren.remove(index);
}

void DiagramItem::setContext(z3::context *context) {m_context = context;}
z3::context *DiagramItem::context() const {return m_context;}

int DiagramItem::resultCount()
{
    return m_results.count();
}

Result *DiagramItem::resultAt(int index)
{
    return m_results[index];
}

void DiagramItem::addResult(QString variable, double result)
{
    Result * newResult = new Result(m_context,this);
    newResult->setVarString(variable);
    newResult->setValNum(result);
    m_results.append(newResult);
}

void DiagramItem::clearResults()
{
    qDeleteAll(m_results);
    m_equationChildren.clear();
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

int DiagramItem::id() const {return childItemNumber();}
int DiagramItem::xPos() const {return m_xPos;}
void DiagramItem::setXPos(int blockXPosition){m_xPos = blockXPosition;}
int DiagramItem::yPos() const {return m_yPos;}
void DiagramItem::setYPos(int blockYPosition){m_yPos = blockYPosition;}

int DiagramItem::type() const
{
    return m_type;
}

void DiagramItem::setType(int itemType)
{
    if (m_type == itemType)
        return;

    m_type = itemType;
}

int DiagramItem::rotation() const
{
    return m_rotation;
}

void DiagramItem::setRotation(int rotation)
{
    m_rotation = rotation;
}

DiagramItem *DiagramItem::thisItem()
{
    return this;
}
