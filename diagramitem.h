//PROVIDES STRUCTURE OF INDIVIDUAL BLOCK ITEMS IN THE MODEL

#ifndef DIAGRAMITEM_H
#define DIAGRAMITEM_H

#include <QObject>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include "z3++.h"
#include "equationparser.h"
#include "equation.h"
#include "port.h"
#include "result.h"
#include "appenums.h"

class DataSource;
class Result;
class DiagramItem : public QObject
{
    Q_OBJECT

    Q_PROPERTY(DiagramItem* thisItem READ thisItem)
    Q_PROPERTY(int id READ id)
    Q_PROPERTY(int xPos READ xPos WRITE setXPos)
    Q_PROPERTY(int yPos READ yPos WRITE setYPos)
    Q_PROPERTY(int numChildren READ childItemCount)
    Q_PROPERTY(int type READ type WRITE setType)
    Q_PROPERTY(int rotation READ rotation WRITE setRotation)

public:
    explicit DiagramItem(int type, z3::context * context, DiagramItem *parentBlock,
                   QObject *parent = nullptr);
    ~DiagramItem();

    // parent
    DiagramItem * parentItem();
    void setParentItem(DiagramItem *parentBlock);

    // Block children
    DiagramItem * childItemAt(int index);
    int IndexOfItem(DiagramItem* childBlock);
    int childItemNumber() const;
    int childItemCount() const;
    void addItemChild(int type, int x, int y);
    void removeItemChild(int modelIndex);

    // Ports
    Port * portAt( int portIndex );
    void addPort(int side, int position);
    void removePort(int portIndex);
    int portCount();

    // Equation children
    int equationCount();
    Equation* equationAt(int index);
    void addEquation();
    void removeEquation(int index);

    // z3 solver and results
    void setContext(z3::context *context);
    z3::context *context() const;
    int resultCount();
    Result* resultAt(int index);
    void addResult(QString variable, double result);
    void clearResults();

    // data save and load
    //    void jsonWrite(QJsonObject &json);
    //    void jsonRead(QJsonObject &json);

    //getters with qProperty
    int id() const;
    int xPos() const;
    int yPos() const;
    DiagramItem* thisItem();

    //setters with qProperty
    void setXPos(int blockXPosition);
    void setYPos(int blockYPosition);

    int type() const;
    void setType(int itemType);

    int rotation() const;
    void setRotation(int rotation);

signals:
    // qAbstractItemModel signals
    void beginResetPorts();
    void endResetPorts();

    void beginInsertPort(int portIndex);
    void endInsertPort();
    void beginRemovePort(int portIndex);
    void endRemovePort();

private:
    //data model pointers
    DiagramItem * m_parentItem;
    QVector<DiagramItem*> m_itemChildren;
    QVector<Port*> m_ports;
    QVector<Equation*> m_equationChildren;
    QVector<Result*> m_results;

    z3::context* m_context;

    //Data
    int m_xPos;
    int m_yPos;
    int m_type;
    int m_rotation;
};

#endif // DIAGRAMITEM_H
