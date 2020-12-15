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
public:
    enum PortSide {
        Top,
        Bottom,
        Left,
        Right
    };

    explicit Port(QObject *parent = nullptr);

    ~Port();

    void setBlockParent(BlockItem * blockParent);

    void setSide(int side);

    void setPosition(int position);

    int side() const;

    int position() const;

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
