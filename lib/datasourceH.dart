import 'diagramitemH.dart';
import 'equationsolverH.dart';

class DataSource
{
    DataSource(var parent);
    ~DataSource();

    DiagramItem * proxyRoot();
    Q_INVOKABLE DiagramItem * proxyChild(var blockIndex);  //Q_INVOKABLE
    Q_INVOKABLE Port * proxyPort(int blockIndex, int portIndex);

    void newProxyRoot(DiagramItem *newProxyRoot);

    Q_INVOKABLE void appendDiagramItem(int type, int x = 0, int y = 0);
    Q_INVOKABLE void deleteDiagramItem(int index);

    Q_INVOKABLE void addEquation();
    Q_INVOKABLE void deleteEquation(int index);

    Q_INVOKABLE void downLevel(int modelIndex);
    Q_INVOKABLE void upLevel();
    Q_INVOKABLE void printFullTree(DiagramItem * rootItem, int depth);
    Q_INVOKABLE void printBlock(int blockIndex);
    Q_INVOKABLE int distanceFromRoot() const;

    //Q_INVOKABLE void addPort( int index, int side, int position );
    Q_INVOKABLE void addPort( int index, QPointF center );
    Q_INVOKABLE void deletePort( int index, int portIndex );

    Q_INVOKABLE void startLink( int index, int portIndex );
    Q_INVOKABLE void deleteLink( int index, int portIndex, int linkIndex );
    Q_INVOKABLE void endLinkFromLink( Link* thisLink );
    Q_INVOKABLE void endLinkFromPort( Port* thisPort );
    Q_INVOKABLE void disconnectPortfromLink( Link* thisLink );
    Q_INVOKABLE void resetLinkstoPort( Port * thisPort);
    Q_INVOKABLE void resetConnectedLinkstoPort( Port * thisPort);

    /* EXPOSING EQUATIONSOLVER FUNCTIONS AS SLOTS TO QML VIA BLOCKDATASOURCE->BLOCKMODEL */
    Q_INVOKABLE void solveEquations();

    /* FUNCTIONS AS SLOTS TO QML TO AID IN QUI OPERATIONS */
    Q_INVOKABLE int maxItemX();
    Q_INVOKABLE int maxItemY();

signals:
    //blocks
    void beginResetDiagramItems();
    void endResetDiagramItems();
    void beginInsertDiagramItem(int blockIndex);
    void endInsertDiagramItem();
    void beginRemoveDiagramItem(int blockIndex);
    void endRemoveDiagramItem();

    void beginResetEquations();
    void endResetEquations();
    void beginInsertEquation(int index);
    void endInsertEquation();
    void beginRemoveEquation(int index);
    void endRemoveEquation();

    void beginResetResults();
    void endResetResults();
    void beginInsertResult(int index);
    void endInsertResult();
    void beginRemoveResult(int index);
    void endRemoveResult();

private:
    DiagramItem * m_root;
    z3::context m_context;
    DiagramItem * m_proxyRoot;
    Port* m_pendingConnectPort;
    Link* m_pendingConnectLink;
}
