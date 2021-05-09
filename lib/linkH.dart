#ifndef LINK_H
#define LINK_H

#include <QObject>
#include <QDebug>
#include <QPainter>
#include "port.h"

class Port; // added to remove circular reference error with blockport.h

class Link : public QObject
{
    Q_OBJECT
public:
    Q_PROPERTY(Link* thisLink READ thisLink)
    Q_PROPERTY(QPointF lastPoint READ lastPoint WRITE setLastPoint NOTIFY lastPointChanged)
    Q_PROPERTY(bool portConnected READ portConnected NOTIFY portConnectedChanged)

    explicit Link(QObject *parent = nullptr);
    ~Link();

    void removePoint(int index);
    void removeLastPoint();
    QPointF lastPoint() const;
    void setLastPoint(QPointF point);

    void setEndPort(Port * endPort);
    void disconnectEndPort();

    Link* thisLink();

    bool portConnected() const;
    Port* startPort();

    void setStartPort(Port *startPort);

signals:
    void lastPointChanged(QPointF newPoint);
    void portConnectedChanged(bool portConnected);

private:
    Port * m_start;
    Port * m_end;
    bool m_portConnected;
    QPointF m_lastPoint;
};

#endif // LINK_H
