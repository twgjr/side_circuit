#ifndef PORT_H
#define PORT_H

#include <QObject>
#include <QDebug>
#include "link.h"
#include "portmodel.h"

class DiagramItem;  //added to remove circular include with blockitem.h
class Link;

class Port : public QObject
{
    Q_OBJECT

public:
    Q_PROPERTY(Port* thisPort READ thisPort)
    Q_PROPERTY(int side READ side WRITE setSide)
    Q_PROPERTY(int position READ position WRITE setPosition)
    Q_PROPERTY(QString name READ name WRITE setName)
    Q_PROPERTY(QPointF absPoint READ absPoint WRITE setAbsPoint NOTIFY absPointChanged)

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
    void resetLinkModel();

    Port* thisPort();

    QPointF absPoint() const;
    void setAbsPoint(QPointF centerPoint);

    //connected links
    QVector<Link*> connectedLinks();
    void appendConnectedLink(Link* cLink);
    void removeConnectedLink(Link* cLink);
    void removeAllLinks();

signals:
    void beginResetPort();
    void endResetPort();
    void beginInsertLink(int linkIndex);
    void endInsertLink();
    void beginRemoveLink(int linkIndex);
    void endRemoveLink();

    void absPointChanged(QPointF absPoint);

private:
    DiagramItem * m_itemParent;
    QVector<Link*> m_links;
    QVector<Link*> m_connectedLinks;

    int m_side;
    int m_position;
    QString m_name;
    QPointF m_absPoint;
};

#endif // PORT_H
