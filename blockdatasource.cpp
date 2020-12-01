#include "blockdatasource.h"

BlockDataSource::BlockDataSource(QObject *parent) : QObject(parent)
{
    qDebug()<<"BlockDataSource object created.";
}

bool BlockDataSource::loadBlockItems(QVariant loadLocation)
{
    QFile loadFile( loadLocation.toUrl().toLocalFile() );

    if (!loadFile.open(QIODevice::ReadOnly)) {
        qWarning("Couldn't open save file.");
        return false;
    }

    clearBlockItems();
    qDebug() << "Cleared current model ";

    QByteArray saveData = loadFile.readAll();
    QJsonDocument loadDoc( QJsonDocument::fromJson(saveData) );
    QJsonObject json = loadDoc.object();

    if (json.contains("Blocks") && json["Blocks"].isArray()) {
        QJsonArray blockArray = json["Blocks"].toArray();

        for ( int i = 0; i < blockArray.size() ; i++){
            QJsonObject jsonobject = blockArray.at(i).toObject();
            BlockItem * blockItem = new BlockItem(&m_context,this);
            blockItem->jsonRead(jsonobject);
            appendBlockItem(blockItem);
            qDebug()<<"Loaded Block# "<<blockItem->id()<<" / "<<blockItem->category()<<" at: "<<blockItem->blockXPosition()<<" x "<<blockItem->blockYPosition()
                   <<" with equation "<<blockItem->equation();
        }
        qDebug() << "Loaded system model";
        return true;
    }else{
        qDebug()<<"Could not load the system model from JSON file!";
        return false;
    }
}

bool BlockDataSource::saveBlockItems(QVariant saveLocation)
{
    QFile saveFile( saveLocation.toUrl().toLocalFile());

    if (!saveFile.open(QIODevice::WriteOnly)) {
        qWarning("Couldn't open save file.");
        return false;
    }

    QJsonArray blockArray;

    qDebug() << "created Block Array";

    for ( int i = 0; i < m_BlockItems.size() ; i++){
        QJsonObject jsonBlockObject;
        m_BlockItems[i]->jsonWrite(jsonBlockObject);
        blockArray.append(jsonBlockObject);
    }

    QJsonObject jsonObject;
    jsonObject["Blocks"]=blockArray;

    saveFile.write( QJsonDocument(jsonObject).toJson() );
    qDebug() << "Saved system model";
    return true;
}

void BlockDataSource::appendBlockItem(BlockItem *blockItem)
{
    emit preItemAdded();
    m_BlockItems.append(blockItem);
    emit postItemAdded();
}

void BlockDataSource::appendBlockItem()
{
    BlockItem * blockItem = new BlockItem(&m_context,this);
    blockItem->setId(m_BlockItems.size());
    appendBlockItem(blockItem);
}

void BlockDataSource::appendBlockItem(const QString &blockItemCategory)
{
    BlockItem * blockItem = new BlockItem(&m_context,this);
    blockItem->setCategory(blockItemCategory);
    blockItem->setId(m_BlockItems.size());
    appendBlockItem(blockItem);
}

void BlockDataSource::appendBlockItem(const QString &blockItemCategory, const int &blockItemId)
{
    BlockItem * blockItem = new BlockItem(&m_context,this);
    blockItem->setCategory(blockItemCategory);
    blockItem->setId(blockItemId);
    appendBlockItem(blockItem);
}

void BlockDataSource::removeBlockItem(int index)
{
    if ( m_BlockItems.size()!=0){
        emit preItemRemoved(index);
        m_BlockItems.removeAt(index);
        emit postItemRemoved();
    }else{
        qDebug()<<"No blocks left to remove!";
    }
}

void BlockDataSource::clearBlockItems()
{
    //for loop has unknown fail at clearing items from model.
    //use while loop to keep trying.
    //set max attempts based on 200% of model size.
    //TODO: implement separate QAbstractListModel function to
    //remove more than one row.
    int maxAttempts = 2 * m_BlockItems.size();
    int attempts = 0;
    while ( m_BlockItems.size() > 0 && attempts < maxAttempts){
        removeLastBlockItem();
        attempts++;
    }
    qDebug()<<"attempts needed to clear model: "<<attempts<<"/"<<maxAttempts/2;
    if(m_BlockItems.size() != 0){
        qDebug()<<"Attemped to clear model, but not all blocks were removed!";
    }else{
        qDebug()<<"All blocks were removed from model";
    }
}

void BlockDataSource::removeLastBlockItem()
{
    if ( !m_BlockItems.isEmpty()){
        removeBlockItem(m_BlockItems.size()-1);
    }
}

QVector<BlockItem *> BlockDataSource::dataItems()
{
    return m_BlockItems;
}

int BlockDataSource::maxBlockX()
{
    int maxVal=0;

    for (int i = 0; i < m_BlockItems.size(); i++){
        int tempVal = m_BlockItems.at(i)->blockXPosition();
        if(tempVal>maxVal){
            maxVal = tempVal;
        }
    }

    return maxVal;
}

int BlockDataSource::maxBlockY()
{
    int maxVal=0;

    for (int i = 0 ; i < m_BlockItems.size(); i++){
        int tempVal = m_BlockItems.at(i)->blockYPosition();
        if(tempVal>maxVal){
            maxVal = tempVal;
        }
    }

    return maxVal;
}



void BlockDataSource::solveEquations()
{
    EquationSolver equationSolver(&m_context);

    //append all equations
    for(int i = 0; i < m_BlockItems.size(); i++)
    {
        //        for(int j = 0; j < m_BlockItems[i]->equationObjList().size(); j++)
        //        {
        //            z3::expr expression = m_BlockItems[i]->equationObjList()[j]->getEquationExpression();
        //            equationSolver.registerEquation(expression);
        //        }

        z3::expr expression = m_BlockItems[i]->equation()->getEquationExpression();
        equationSolver.registerEquation(expression);
    }

    //solve
    equationSolver.solveEquations();

    //set solution to return to app
}
