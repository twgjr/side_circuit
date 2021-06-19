import 'dart:math';

import 'package:flutter/material.dart';

import 'port.dart';
import 'link.dart';
import 'equation.dart';
import 'result.dart';
import 'model.dart';

class Diagram {
    DiagramItem root;
    DiagramItem proxyRoot;
    Port pendingConnectPort;
    Link pendingConnectLink;
    Model model = Model();


    Diagram() {
        //by default the root node is always a block
        root =  DiagramItem.root(this.model);  // real root is empty block
        proxyRoot = root; // always start at empty top level
    }

    void setProxyRoot(DiagramItem dItem) {
        proxyRoot = dItem;
    }

    void moveUp(){
        if (proxyRoot.parent != null) {
            setProxyRoot(proxyRoot.parent);
        }
        //root.printTree(root);
    }

    void moveDown(DiagramItem dItem){
        setProxyRoot(dItem);
        //root.printTree(root);
    }

    void moveToTop() {
        proxyRoot = root;
        //root.printTree(root);
    }

    void solve() {
        print("solving");
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
}

class DiagramItem {
    //data model pointers
    DiagramItem  parent;
    List<DiagramItem> children = [];
    List<Port> ports = [];
    List<Equation> equations = [];
    List<Result> results = [];
    Model model;

    //Data
    double xPosition = 0;
    double yPosition = 0;
    int type = 0;
    int rotation = 0;

    DiagramItem(this.model,this.type, this.parent);

    DiagramItem.root(this.model);

    DiagramItem.child(this.model,this.parent) {
        this.type = 0;
        //this.equations.add(Equation.string("test"));
    }

    int breadth() {
        if (this.parent!=null) {
            return this.parent.children.indexOf(this);
        } else {
            return 0;
        }
    }

    int depth() {
        int count = 0;

        if(this.parent==null){
            return count; //at the real root
        }

        DiagramItem nextItem = this.parent;
        count+=1;

        while(nextItem.parent!=null) {
            nextItem = nextItem.parent;
            count+=1;
        }
        return count;
    }

    void addChild() {
        DiagramItem child = DiagramItem.child(this.model,this);
        this.children.add(child);
        //printTree(getRoot());
    }

    void deleteChild(DiagramItem child) {
        this.children.remove(child);
        //printTree(getRoot());
    }

    void addPort(Point center) {
        Port  newPort = Port();
        newPort.itemParent = this;
        newPort.absPoint = center;
        ports.add(newPort);
    }

    void addEquation() {
        Equation  newEquation = Equation(this.model);
        equations.add(newEquation);
    }

    void addEquationString(String equationString) {
        Equation  newEquation = Equation.string(this.model,equationString);
        equations.add(newEquation);
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
            dItem = dItem.parent;
        }
        return dItem;
    }

    void printTree(DiagramItem dItem) {
        //breadth first print children
        String spacer = "";
        for ( int ctr = 0 ; ctr < dItem.depth() ; ctr++) {
            spacer += "->";
        }

        print("$spacer${dItem.depth()},${dItem.breadth()}");

        for ( DiagramItem child in dItem.children) {
            printTree(child);
        }
    }
}