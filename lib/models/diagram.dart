import 'dart:math';

import 'port.dart';
import 'link.dart';
import 'equation.dart';
import 'result.dart';

class Diagram {
    DiagramItem root;
    DiagramItem proxyRoot;
    Port pendingConnectPort;
    Link pendingConnectLink;

    Diagram() {
        //by default the root node is always a block
        root =  DiagramItem.root();  // real root is empty block
        proxyRoot = root; // always start at empty top level
    }

    int distanceFromRoot() {
        int count = 0;

        if(proxyRoot.parent==null){
            return count; //at the real root
        }

        DiagramItem realItem = proxyRoot;
        realItem = realItem.parent;
        count+=1;

        while(realItem.parent!=null) {
            realItem = realItem.parent;
            count+=1;
        }
        return count;
    }

    void endLinkFromLink( Link link ) {
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

    void endLinkFromPort( Port port ) {
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

    void disconnectPortfromLink(Link link) {
        link.disconnectEndPort();
    }

    void solveEquations() {
    }

    double maxItemX() {
        double blockX = 0;
        for ( int i = 0 ; i < proxyRoot.children.length ; i++ ) {
            double newBlockX = proxyRoot.children[i].xPosition;
            if(blockX<newBlockX){
                blockX = newBlockX;
            }
        }
        return blockX;
    }

    double maxItemY() {
        double blockY = 0;
        for ( int i = 0 ; i < proxyRoot.children.length ; i++ ) {
            double newBlockX = proxyRoot.children[i].yPosition;
            if(blockY<newBlockX){
                blockY = newBlockX;
            }
        }
        return blockY;
    }
}

class DiagramItem {
    //data model pointers
    DiagramItem  parent;
    List<DiagramItem> children = [];
    List<Port> ports = [];
    List<Equation> equations = [];
    List<Result> results = [];

    //Data
    double xPosition = 0;
    double yPosition = 0;
    int type = 0;
    int rotation = 0;

    DiagramItem(this.type, this.parent);

    DiagramItem.root();

    DiagramItem.child(this.parent) {
        this.type = 0;
        this.equations.add(Equation.string("test"));
    }

    int indexOfItem(DiagramItem childBlock) {
        return children.indexOf(childBlock);
    }

    int childItemNumber() {
        if (parent!=null) {
            return parent.children.indexOf(this);
        } else {
            return 0;
        }
    }

    void addPort(Point center) {
        Port  newPort = Port();
        newPort.itemParent = this;
        newPort.absPoint = center;
        ports.add(newPort);
    }

    void addEquation() {
        Equation  newEquation = Equation();
        equations.add(newEquation);
    }

    void addResult(String name, num value) {
        Result  newResult = Result();
        newResult.name = name;
        newResult.value = value;
        results.add(newResult);
    }
}