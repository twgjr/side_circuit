#include "equationmodel.h"

EquationModel::EquationModel(QObject *parent) : QAbstractItemModel(parent),
    m_signalConnected(false)
{
    //qDebug()<<"Created: "<<this<<" with Qparent: "<<parent;

    //set the basic roles for access of item properties in QML
    m_roles[ProxyRoot]="proxyRoot";
    m_roles[ThisRole]="thisItem";
    m_roles[TypeRole]="type";
    m_roles[DescRole]="description";
    m_roles[IdRole]="id";
    m_roles[XposRole]="xPos";
    m_roles[YposRole]="yPos";
    m_roles[WidthRole]="itemWidth";
    m_roles[HeightRole]="itemHeight";
    m_roles[EquationRole]="equationString";
}

QModelIndex EquationModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)){
        return QModelIndex();
    }

    Equation *childEquationItem = m_dataSource->proxyRoot()->equationAt(row);
    return createIndex(row, column, childEquationItem);

    return QModelIndex();
}

QModelIndex EquationModel::parent(const QModelIndex &index) const
{
    Q_UNUSED(index);
    return QModelIndex();
}

int EquationModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0){
        return 0;
    }

    int equationCount = m_dataSource->proxyRoot()->equationCount();

    return equationCount;
}

int EquationModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1; //columns not used
}

QVariant EquationModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()){
        return QVariant();
    }

    Equation * item = m_dataSource->proxyRoot()->equationAt(index.row());
    QByteArray roleName = m_roles[role];
    QVariant name = item->property(roleName.data());
    return name;

    return QVariant();
}

bool EquationModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    bool somethingChanged = false;

    Equation * equationItem = m_dataSource->proxyRoot()->equationAt(index.row());
    switch (role) {
    case EquationRole:
        if(equationItem->equationString() != value.toString()){
            equationItem->setEquationString(value.toString());
        }
        break;
    }

    if(somethingChanged){
        emit dataChanged(index, index, QVector<int>() << role);
        return true;
    }
    return false;
}

Qt::ItemFlags EquationModel::flags(const QModelIndex &index) const
{
    if (!index.isValid()){
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}

QHash<int, QByteArray> EquationModel::roleNames() const
{
    return m_roles;
}

DataSource *EquationModel::dataSource() const
{
    return m_dataSource;
}

void EquationModel::setdataSource(DataSource *newDataSource)
{
    beginResetModel();
    if(m_dataSource && m_signalConnected){
        m_dataSource->disconnect(this);
    }

    m_dataSource = newDataSource;

    connect(m_dataSource,&DataSource::beginResetEquations,this,[=](){
        beginResetModel();
    });
    connect(m_dataSource,&DataSource::endResetEquations,this,[=](){
        endResetModel();
    });
    connect(m_dataSource,&DataSource::beginInsertEquation,this,[=](int index){
        beginInsertRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endInsertEquation,this,[=](){
        endInsertRows();
    });
    connect(m_dataSource,&DataSource::beginRemoveEquation,this,[=](int index){
        beginRemoveRows(QModelIndex(),index,index);
    });
    connect(m_dataSource,&DataSource::endRemoveEquation,this,[=](){
        endRemoveRows();
    });

    m_signalConnected = true;
    endResetModel();
}
