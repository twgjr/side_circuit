import 'dart:math';
import 'port.dart';
import 'link.dart';
import 'result.dart';

class Diagram {
    DiagramItem? root;
    DiagramItem? proxyRoot;
    Port? pendingConnectPort;
    Link? pendingConnectLink;

    Diagram() {
        //by default the root node is always a block
        root =  DiagramItem.root();  // real root is empty block
        proxyRoot = root; // always start at empty top level
    }

    void setProxyRoot(DiagramItem dItem) {
        proxyRoot = dItem;
    }

    void moveUp(){
        if (proxyRoot!.parent != null) {
            setProxyRoot(proxyRoot!.parent!);
        }
    }

    void moveDown(DiagramItem dItem){
        setProxyRoot(dItem);
    }

    void moveToTop() {
        proxyRoot = root;
    }

    void solve() {
        print("solver not implemented yet");
    }

    void endLinkFromLink( Link link ) {
        if(pendingConnectPort!=null){
            pendingConnectPort!.connectedLinks.add(link);
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
            port.connectedLinks.add(pendingConnectLink!);
            pendingConnectLink!.end = port;
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
}

class DiagramItem {
    //data model pointers
    DiagramItem?  parent;
    List<DiagramItem> children = [];
    List<Port> ports = [];
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
    }

    int breadth() {
        if (this.parent!=null) {
            return this.parent!.children.indexOf(this);
        } else {
            return 0;
        }
    }

    int depth() {
        int count = 0;

        if(this.parent==null){
            return count; //at the real root
        }

        DiagramItem nextItem = this.parent!;
        count+=1;

        while(nextItem.parent!=null) {
            nextItem = nextItem.parent!;
            count+=1;
        }
        return count;
    }

    void addChild() {
        DiagramItem child = DiagramItem.child(this);
        this.children.add(child);
    }

    void deleteChild(DiagramItem child) {
        this.children.remove(child);
    }

    void addPort(Point center) {
        Port  newPort = Port();
        newPort.itemParent = this;
        newPort.absPoint = center;
        ports.add(newPort);
    }

    void addEquationString(String equationString) {
        print("not implemented yet");
    }

    void addResult(String name, num value) {
        Result  newResult = Result();
        newResult.name = name;
        newResult.value = value;
        results.add(newResult);
    }

    DiagramItem getRoot(){
        DiagramItem dItem = this;
        while(dItem.parent != null) {
            dItem = dItem.parent!;
        }
        return dItem;
    }
}