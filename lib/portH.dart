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
    Q_PROPERTY(QString name READ name WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(QPointF absPoint READ absPoint WRITE setAbsPoint NOTIFY absPointChanged)
    Q_PROPERTY(bool linkIsValid READ linkIsValid WRITE setLinkIsValid NOTIFY linkIsValidChanged)

    explicit Port(QObject *parent = nullptr);
    ~Port();

    // this port
    void setItemParent(DiagramItem * blockParent);
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

    bool linkIsValid() const
    {
        return m_linkIsValid;
    }

    void setLinkIsValid(bool connectionIsValid)
    {
        if (m_linkIsValid == connectionIsValid)
            return;

        m_linkIsValid = connectionIsValid;
        emit linkIsValidChanged(m_linkIsValid);
    }

signals:
    void beginResetPort();
    void endResetPort();
    void beginInsertLink(int linkIndex);
    void endInsertLink();
    void beginRemoveLink(int linkIndex);
    void endRemoveLink();

    void absPointChanged(QPointF absPoint);
    void linkIsValidChanged(bool connectionIsValid);
    void nameChanged(QString name);

private:
    DiagramItem * m_itemParent;
    QVector<Link*> m_links;
    QVector<Link*> m_connectedLinks;

    QString m_name;
    QPointF m_absPoint;
    bool m_linkIsValid;
};

#endif // PORT_H
