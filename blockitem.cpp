#include "blockitem.h"



BlockItem::BlockItem(z3::context *context, QObject *parent) : QObject(parent),
    m_category(""),
    m_id(0),
    m_blockXPosition(0),
    m_blockYPosition(0),
    m_equation(context),
    m_solverContextReference(context)
{
    qDebug()<<"Block Item created.";
}

void BlockItem::jsonRead(QJsonObject &json)
{
    if (json.contains("category") && json["category"].isString()){
        m_category = json["category"].toString();
    }else{
        qDebug()<<"Could not load category";
    }

    if (json.contains("Block_ID") && json["Block_ID"].isDouble()){
        m_id = json["Block_ID"].toInt();
    }else{
        qDebug()<<"Could not load id";
    }

    if (json.contains("BlockXPosition") && json["BlockXPosition"].isDouble()){
        m_blockXPosition = json["BlockXPosition"].toInt();
    }else{
        qDebug()<<"Could not load BlockXPosition";
    }

    if (json.contains("BlockYPosition") && json["BlockYPosition"].isDouble()){
        m_blockYPosition = json["BlockYPosition"].toInt();
    }else{
        qDebug()<<"Could not load BlockYPosition";
    }
    if (json.contains("Equation") && json["Equation"].isString()){
        m_equation.setEquationString(json["Equation"].toString());
    }else{
        qDebug()<<"Could not load Equation";
    }
}

void BlockItem::jsonWrite(QJsonObject &json)
{
    json["Block_ID"] = m_id;
    json["category"] = m_category;
    json["BlockXPosition"] = m_blockXPosition;
    json["BlockYPosition"] = m_blockYPosition;
    json["Equation"] = m_equation.getEquationString();
    //Commented code sample for how to add an array
    /*
    QJsonArray npcArray;
    for (const Character &npc : mNpcs) {
        QJsonObject npcObject;
        npc.write(npcObject);
        npcArray.append(npcObject);
    }
    json["npcs"] = npcArray;
    */
}

QString BlockItem::category() const {return m_category;}
void BlockItem::setCategory(QString category)
{
    m_category = category;
    emit categoryChanged(m_category);
}

int BlockItem::id() const {return m_id;}
void BlockItem::setId(int id)
{
    m_id = id;
    emit idChanged(m_id);
}

int BlockItem::blockXPosition() const {return m_blockXPosition;}
void BlockItem::setBlockXPosition(int blockXPosition)
{
    m_blockXPosition = blockXPosition;
    emit blockXPositionChanged(m_blockXPosition);
}

int BlockItem::blockYPosition() const {return m_blockYPosition;}
void BlockItem::setBlockYPosition(int blockYPosition)
{
    m_blockYPosition = blockYPosition;
    emit blockYPositionChanged(m_blockYPosition);
}

Equation * BlockItem::equation(){return &m_equation;}
void BlockItem::setEquation(QString equationString)
{
    m_equation.setEquationString(equationString);
    emit equationChanged(equationString);
}
