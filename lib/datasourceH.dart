import 'dart:math';

import 'diagramitemH.dart';
import 'portH.dart';
import 'linkH.dart';

class DataSource
{
    DiagramItem root;
    DiagramItem proxyRoot;
    Port pendingConnectPort;
    Link pendingConnectLink;

    DataSource()
    {
        //by default the root node is always a block
        root =  DiagramItem(0,null);  // real root is empty block
        proxyRoot = root; // always start at empty top level
    }

    void appendDiagramItem(int type, int x, int y)
    {
        proxyRoot.addItemChild(type,x,y);
    }

    void addEquation()
    {
        proxyRoot.addEquation();
    }

    void deleteEquation(int index)
    {
        proxyRoot.equations.removeAt(index);
    }

    void newProxyRoot(DiagramItem newProxyRoot)
    {
        proxyRoot=newProxyRoot;
    }

    void downLevel(int index)
    {
        newProxyRoot(proxyRoot.chidren[index]);
    }

    void upLevel()
    {
        if(proxyRoot.parent!=null){
            // parent is valid, cannot go higher than actual root
            newProxyRoot(proxyRoot.parent);
        }
    }

    int distanceFromRoot()
    {
        int count = 0;

        if(proxyRoot.parent==null){
            return count; //at the real root
        }

        DiagramItem realItem = proxyRoot;
        realItem = realItem.parent;
        count+=1;

        while(realItem.parent!=null){
            realItem = realItem.parent;
            count+=1;
        }
        return count;
    }

    void addPort(int index, Point center)
    {
        proxyRoot.chidren[index].addPort(center);
    }

    void deletePort(int index, int portIndex)
    {
        proxyRoot.chidren[index].ports.removeAt(portIndex);
    }

    void startLink(int index, int portIndex)
    {
        proxyRoot.chidren[index].ports[portIndex].startLink();
    }

    void deleteLink(int index, int portIndex, int linkIndex)
    {
        proxyRoot.chidren[index].ports[portIndex].links.removeAt(linkIndex);
    }

    void endLinkFromLink( Link link )
    {
        if(pendingConnectPort!=null){
            pendingConnectPort.connectedLinks.add(link);
            link.end = pendingConnectPort;
            //cleanup the buffer pointers when done connecting
            pendingConnectPort = null;
            pendingConnectLink = null;
        } else {
            pendingConnectLink = link;
        }
    }

    void endLinkFromPort( Port port )
    {
        if(pendingConnectLink != null){
            port.connectedLinks.add(pendingConnectLink);
            pendingConnectLink.end = port;
            //cleanup the buffer pointers when done connecting
            pendingConnectPort = null;
            pendingConnectLink = null;
        } else {
            pendingConnectPort = port;
        }
    }

    void disconnectPortfromLink(Link link)
    {
        link.disconnectEndPort();
    }

    void solveEquations()
    {
    }

    int maxItemX()
    {
        int blockX = 0;
        for ( int i = 0 ; i < proxyRoot.chidren.length ; i++ ) {
            int newBlockX = proxyRoot.chidren[i].xPosition;
            if(blockX<newBlockX){
                blockX = newBlockX;
            }
        }
        return blockX;
    }

    int maxItemY()
    {
        int blockY = 0;
        for ( int i = 0 ; i < proxyRoot.chidren.length ; i++ ) {
    int newBlockX = proxyRoot.chidren[i].yPosition;
    if(blockY<newBlockX){
    blockY = newBlockX;
    }
    }
        return blockY;
    }
}
