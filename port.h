#ifndef PORT_H
#define PORT_H

#include <QObject>
#include <QDebug>
#include "link.h"
#include "portmodel.h"

class Block;  //added to remove circular include with blockitem.h
class Element;

class Port : public QObject
{
    Q_OBJECT

public:
    Q_PROPERTY(Port* thisPort READ thisPort WRITE setThisPort NOTIFY thisPortChanged)
    //Q_PROPERTY(Link* connectedLink READ connectedLink WRITE setConnectedLink NOTIFY connectedLinkChanged)

    Q_PROPERTY(int side READ side WRITE setSide)
    Q_PROPERTY(int position READ position WRITE setPosition)
    Q_PROPERTY(QString name READ name WRITE setName)
    Q_PROPERTY(int state READ state WRITE setState NOTIFY stateChanged)

    explicit Port(QObject *parent = nullptr);
    ~Port();

    // this port
    void setBlockParent(Block * blockParent);
    void setElementParent(Element *elementParent);
    int side() const;
    void setSide(int side);
    int position() const;
    void setPosition(int position);
    QString name() const;
    void setName(QString name);

    //links
    QVector<Link *> links();
    Link * linkAt(int linkIndex);
    int linkCount();
    void startLink();
    void removeLink(int linkIndex);
    //Link* connectedLink(int index);
    void setConnectedLink(Link* connectedLink);

    Port* thisPort();
    void setThisPort(Port* thisPort);

    int state() const;
    void setState(int state);


signals:
    void beginResetPort();
    void endResetPort();
    void beginInsertLink(int linkIndex);
    void endInsertLink();
    void beginRemoveLink(int linkIndex);
    void endRemoveLink();

    void thisPortChanged(Port* thisPort);
    void connectedLinkChanged(Link* connectedLink);

    void stateChanged(int state);

private:
    Block * m_blockParent;
    Element * m_elementParent;
    int m_side;
    int m_position;
    QString m_name;
    bool isConnected;
    int m_id;

    QVector<Link*> m_links;  //links started at this port
    Port* m_thisPort;
    QVector<Link*> m_connectedLinks; //links ending at this port
    int m_state;
};

#endif // PORT_H
