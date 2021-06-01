import 'dart:math';

import 'equationH.dart';
import 'portH.dart';
import 'resultH.dart';

class DiagramItem {
    //data model pointers
    DiagramItem  parent;
    List<DiagramItem> chidren;
    List<Port> ports;
    List<Equation> equations;
    List<Result> results;

    //Data
    int xPosition;
    int yPosition;
    int type;
    int rotation;

    DiagramItem(int type, DiagramItem parentItem);

    int indexOfItem(DiagramItem childBlock)
    {
        return chidren.indexOf(childBlock);
    }

    int childItemNumber() {
        if (parent!=null) {
            return parent.chidren.indexOf(this);
        } else {
            return 0;
        }
    }

    void addItemChild(int type, int x, int y)
    {
        DiagramItem childItem = DiagramItem(type,this);
        childItem.xPosition=x;
        childItem.yPosition=y;
        chidren.add(childItem);
    }

    void addPort(Point center)
    {
        Port  newPort = Port();
        newPort.itemParent = this;
        newPort.absPoint = center;
        ports.add(newPort);
    }

    void addEquation()
    {
        Equation  newEquation = Equation();
        equations.add(newEquation);
    }

    void addResult(String name, num value)
    {
        Result  newResult = Result();
        newResult.name = name;
        newResult.value = value;
        results.add(newResult);
    }
}
