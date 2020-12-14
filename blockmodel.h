// ORGANIZES THE BLOCKS IN A FORMAT USABLE BY LISTVIEW IN QML
// SHOULD ONLY CONTAIN THINGS ASSOCIATED WITH THE QML/C++ INTERFACE
// SOME FUNCTIONS MAY BE REPEATED FROM OTHER CLASSES TO CLEANLY
// EXPOSE THEM TO QML. BETTER FOR READABILITY AND SIMPLICITY

#ifndef BLOCKMODEL_H
#define BLOCKMODEL_H

#include <QObject>
#include <QAbstractItemModel>
#include <QDebug>
#include <QFile>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include "blockitem.h"
#include "equationsolver.h"

class BlockModel : public QAbstractItemModel
{
    Q_OBJECT
    Q_PROPERTY(QVariantList roles READ roles WRITE setRoles NOTIFY rolesChanged)

public:
    explicit BlockModel(//const QStringList &headers,
                        QObject *parent = nullptr);
    ~BlockModel();

    // QAbstractItemModel read-only functions
    QVariant data(const QModelIndex &index, int role) const override;
    QModelIndex index(int row,
                      int column,
                      const QModelIndex &parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex &index) const override;
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;


    // QAbstractItemModel functions for editable model
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    bool setData(const QModelIndex &index,
                 const QVariant &value,
                 int role) override;
//    bool insertRows(int position,
//                    int rows,
//                    const QModelIndex &parent = QModelIndex()) override;
//    bool removeRows(int position,
//                    int rows,
//                    const QModelIndex &parent = QModelIndex()) override;

    //functions for working with roles
    QHash<int,QByteArray> roleNames() const override;
    QVariantList roles() const;
    void setRoles(QVariantList roles);

    QModelIndex qIndexOfBlock(BlockItem *item);
    BlockItem * blockFromQIndex(const QModelIndex &index) const;

    void newProxyRoot(BlockItem *newProxyRoot);
    void cloneItemComplete(BlockItem * newItem, BlockItem * oldItem);
    void cloneItemData(BlockItem * newItem, BlockItem * oldItem);
    Q_INVOKABLE void appendBlock(int x = 0, int y = 0);
    Q_INVOKABLE void downLevel(int modelIndex);
    Q_INVOKABLE void upLevel();
    Q_INVOKABLE void printProxyTree(BlockItem * parentItem, int depth);
    Q_INVOKABLE void printFullTree(BlockItem * rootItem, int depth);
    Q_INVOKABLE void printBlock(int modelIndex);
    Q_INVOKABLE int distanceFromRoot() const;
    Q_INVOKABLE int numChildren(int modelIndex);
    Q_INVOKABLE void deleteBlock(int modelIndex);
    Q_INVOKABLE void addPort(int modelIndex, int side, int position);

    /* EXPOSING EQUATIONSOLVER FUNCTIONS AS SLOTS TO QML VIA BLOCKDATASOURCE->BLOCKMODEL */
    Q_INVOKABLE void solveEquations();

    /* FUNCTIONS AS SLOTS TO QML TO AID IN QUI OPERATIONS */
    Q_INVOKABLE int maxBlockX();
    Q_INVOKABLE int maxBlockY();

signals:
    void rolesChanged();

private:
    //void setupModelData(const QStringList &lines, BlockItem *parent);
    BlockItem * getItemFromQIndex(const QModelIndex &index) const;
    BlockItem * m_root;
    QHash<int, QByteArray> m_roles;
    z3::context m_context;
    //proxy model to display in GUI; only one level of children at a time
    BlockItem * m_proxyRoot; //proxy model is slow, need to implement show/hide for source model instead, but works for now
};

#endif // BLOCKMODEL_H
