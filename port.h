#ifndef PORT_H
#define PORT_H

#include <QObject>
#include <QDebug>
#include "link.h"
#include "portmodel.h"

class DiagramItem;  //added to remove circular include with blockitem.h

class Port : public QObject
{
    Q_OBJECT

public:
    Q_PROPERTY(Port* thisPort READ thisPort)
    Q_PROPERTY(int side READ side WRITE setSide)
    Q_PROPERTY(int position READ position WRITE setPosition)
    Q_PROPERTY(QString name READ name WRITE setName)

    explicit Port(QObject *parent = nullptr);
    ~Port();

    // this port
    void setItemParent(DiagramItem * blockParent);
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

    Port* thisPort();

signals:
    void beginResetPort();
    void endResetPort();
    void beginInsertLink(int linkIndex);
    void endInsertLink();
    void beginRemoveLink(int linkIndex);
    void endRemoveLink();

    void connectedLinkChanged(Link* connectedLink);

private:
    DiagramItem * m_itemParent;
    QVector<Link*> m_links;
    bool isConnected;

    int m_side;
    int m_position;
    QString m_name;
};

#endif // PORT_H
