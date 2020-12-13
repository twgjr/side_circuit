#ifndef LINK_H
#define LINK_H

#include <QObject>
#include <QDebug>
#include <QPainter>

class Port; // added to remove circular reference error with blockport.h

class Link : public QObject
{
    Q_OBJECT
public:
    explicit Link(QObject *parent = nullptr);

signals:

private:
    Port * m_start;
    Port * m_end;
    QVector<QPoint> points;
};

#endif // LINK_H
