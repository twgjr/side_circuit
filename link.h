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
    Q_PROPERTY(int startX READ startX WRITE setStartX)
    Q_PROPERTY(int startY READ startY WRITE setStartY)
    Q_PROPERTY(int endX READ endX WRITE setEndX)
    Q_PROPERTY(int endY READ endY WRITE setEndY)

    explicit Link(QObject *parent = nullptr);
    ~Link();

    int startX() const;
    int startY() const;
    int endX() const;
    int endY() const;

    void setStartX(int startX);
    void setStartY(int startY);
    void setEndX(int endX);
    void setEndY(int endY);

signals:

private:
    Port * m_start;
    Port * m_end;
    QVector<QPoint> points;
    int m_startX;
    int m_startY;
    int m_endX;
    int m_endY;
};

#endif // LINK_H
