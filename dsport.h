#ifndef DSPORT_H
#define DSPORT_H

#include <QObject>
#include "dschildblock.h"

class DSPort : public QObject
{
    Q_OBJECT

public:
    Q_PROPERTY(Port* dsPort READ dsPort WRITE setdsPort NOTIFY dsPortChanged)

    explicit DSPort(QObject *parent = nullptr);
    ~DSPort();

    Port* dsPort() const;
    Q_INVOKABLE void setdsPort(Port* parentPort);

    Q_INVOKABLE void startLink();
    Q_INVOKABLE void deleteLink(int linkIndex);

signals:
    void dsPortChanged(Port* parentPort);

    void beginResetLinkModel();
    void endResetLinkModel();
    void beginInsertLink(int portIndex);
    void endInsertLink();
    void beginRemoveLink(int portIndex);
    void endRemoveLink();

private:
    Port* m_dsPort;
};

#endif // DSPORT_H
