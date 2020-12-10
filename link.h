#ifndef LINK_H
#define LINK_H

#include <QObject>

class Link : public QObject
{
    Q_OBJECT
public:
    explicit Link(QObject *parent = nullptr);

signals:

};

#endif // LINK_H
