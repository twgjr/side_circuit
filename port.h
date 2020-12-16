#ifndef PORT_H
#define PORT_H

#include <QObject>
#include <QDebug>
//#include "blockitem.h"
#include "link.h"

class BlockItem;  //added to remove circular include with blockitem.h

class Port : public QObject
{
    Q_OBJECT

    Q_PROPERTY(int side READ side WRITE setSide)
    Q_PROPERTY(int position READ position WRITE setPosition)
    Q_PROPERTY(QString name READ name WRITE setName)

public:

    explicit Port(QObject *parent = nullptr);
    ~Port();

    void setBlockParent(BlockItem * blockParent);
    int side() const;
    void setSide(int side);
    int position() const;
    void setPosition(int position);
    QString name() const;
    void setName(QString name);

signals:

private:
    BlockItem * m_blockParent;
    Link * m_link;
    int m_side;
    int m_position;
    QString m_name;
    bool isConnected;
};

#endif // PORT_H
