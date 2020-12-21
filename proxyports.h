#ifndef PROXYPORTS_H
#define PROXYPORTS_H

#include <QObject>
#include "datasource.h"

class ProxyPorts : public QObject
{
    Q_OBJECT
public:
    Q_PROPERTY(BlockItem* parentBlock READ parentBlock WRITE setParentBlock)

    explicit ProxyPorts(QObject *parent = nullptr);

    BlockItem *parentBlock();
    Q_INVOKABLE void setParentBlock(BlockItem* block);

    Q_INVOKABLE void addPort(int side, int position);
    Q_INVOKABLE void deletePort(int portIndex);

signals:
    void parentBlockDSChanged(DataSource* parentBlockDS);

    void beginResetPortModel();
    void endResetPortModel();
    void beginInsertPort(int blockIndex);
    void endInsertPort();
    void beginRemovePort(int blockIndex);
    void endRemovePort();

private:
    BlockItem* m_parentBlock;
};

#endif // PROXYPORTS_H
