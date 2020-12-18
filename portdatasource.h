#ifndef PORTDATASOURCE_H
#define PORTDATASOURCE_H

#include <QObject>
#include "blockdatasource.h"

class PortDataSource : public QObject
{
    Q_OBJECT
public:
    Q_PROPERTY(BlockItem* parentBlock READ parentBlock WRITE setParentBlock)

    explicit PortDataSource(QObject *parent = nullptr);

    BlockItem *parentBlock();
    Q_INVOKABLE void setParentBlock(BlockItem* block);

    Q_INVOKABLE void addPort(int side, int position);
    Q_INVOKABLE void deletePort(int portIndex);
    /*
    Q_INVOKABLE int portCount();
    Q_INVOKABLE int portSide(int portNum);
    Q_INVOKABLE int portPosition(int portNum);
    */
signals:
    void parentBlockDSChanged(BlockDataSource* parentBlockDS);

    void beginResetPortModel();
    void endResetPortModel();
    void beginInsertPort(int blockIndex);
    void endInsertPort();
    void beginRemovePort(int blockIndex);
    void endRemovePort();

private:
    BlockItem* m_parentBlock;
};

#endif // PORTDATASOURCE_H
