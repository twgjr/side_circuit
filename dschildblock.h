#ifndef DSCHILDBlOCK_H
#define DSCHILDBlOCK_H

#include <QObject>
#include "datasource.h"

class DSChildBlock : public QObject
{
    Q_OBJECT
public:
    Q_PROPERTY(BlockItem* dsChildBlock READ dsChildBlock WRITE setdsChildBlock NOTIFY dsChildBlockChanged)

    explicit DSChildBlock(QObject *parent = nullptr);
    ~DSChildBlock();

    //blocks
    BlockItem *dsChildBlock();
    Q_INVOKABLE void setdsChildBlock(BlockItem* block);

    //ports
    Q_INVOKABLE Port * port(int portIndex);
    Q_INVOKABLE void addPort(int side, int position);
    Q_INVOKABLE void deletePort(int portIndex);

    //links
    Q_INVOKABLE void startLink(int portIndex);

signals:
    void beginResetPortModel();
    void endResetPortModel();
    void beginInsertPort(int portIndex);
    void endInsertPort();
    void beginRemovePort(int portIndex);
    void endRemovePort();

    void dsChildBlockChanged(BlockItem* dsChildBlock);

private:
    BlockItem* m_dsChildBlock;
};

#endif // DSCHILDBlOCK_H
