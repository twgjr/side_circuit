/*
DIAGRAM adds a layer of abstraction between the circuit and the UI.  It prepares the circuit
for viewing, editing, and running the underlying simulator from the UI.  A SUBCIRCUIT is a 
collection of elements and other subcircuits.  At the highest level the root is a SUBCIRCUIT.
If the root is the only subcircuit then it is the top level contains only elements.
*/

import 'dart:math';
import 'circuits.dart';

enum DiagramItemKind{
  SUBCIRCUIT,
  ELEMENT,
}

class Diagram {
    DiagramItem? root;
    DiagramItem? subCircuit;

    Diagram() {
        //by default the root node is always a block
        root =  DiagramItem.root();  // real root is empty block
        subCircuit = root; // always start at empty top level
    }

    void setProxyRoot(DiagramItem dItem) {
        subCircuit = dItem;
    }

    void moveUp(){
        if (subCircuit!.parent != null) {
            setProxyRoot(subCircuit!.parent!);
        }
    }

    void moveDown(DiagramItem dItem){
        setProxyRoot(dItem);
    }

    void moveToTop() {
        subCircuit = root;
    }

    void solve() {
        print("solver not implemented yet");
    }
}

class DiagramItem {
  /*
  DiagramItem can be either a subcircuit or an element.  Elements also have their own kinds 
  which are defined in circuits.dart under the ElementKind enum
  */
    DiagramItem?  parent;
    List<DiagramItem> children = [];
    DiagramItemKind kind = DiagramItemKind.SUBCIRCUIT;
    List<Point<double>> terminals = [];

    double xPosition = 0;
    double yPosition = 0;
    int rotation = 0;

    DiagramItem.root();

    DiagramItem(this.parent, this.kind, this.xPosition, this.yPosition, this.rotation) {
    // Add a default terminal at the center of the item
    terminals.add(Point<double>(0, 0));
    }

    int breadth() {
        if (this.parent!=null) {
            return this.parent!.children.indexOf(this);
        } else {
            return 0;
        }
    }

    void addDiagramItem() {
        DiagramItem dItem = DiagramItem(this, DiagramItemKind.SUBCIRCUIT, 0, 0, 0);
        this.children.add(dItem);
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

    void remove(DiagramItem child) {
        this.children.remove(child);
    }

    DiagramItem getRoot(){
        DiagramItem dItem = this;
        while(dItem.parent != null) {
            dItem = dItem.parent!;
        }
        return dItem;
    }
}