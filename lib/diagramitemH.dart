//PROVIDES STRUCTURE OF INDIVIDUAL BLOCK ITEMS IN THE MODEL

//#include "z3++.h"
//import 'parser.dart';
import 'equationH.dart';
import 'portH.dart';
import 'resultH.dart';
//import 'appenumsH.dart';

class DiagramItem
{
    DiagramItem(int type, /*z3::context context,*/ DiagramItem parentItem);

    //data model pointers
    DiagramItem  parentItem;
    var items = [];  //QVector<DiagramItem>
    var ports = []; //QVector<Port>
    var equations = []; //QVector<Equation>
    var results = [];  //QVector<Result>

    //z3::context _m_context;

    //Data
    int xPosition;
    int yPosition;
    int type;
    int rotation;

    int indexOfItem(DiagramItem childBlock)
    {
        return items.indexOf(childBlock);
    }

    int childItemNumber() {
        if (parentItem!=null) {
            return parentItem.items.indexOf(this);
        } else {
            return 0;
        }
    }

    void addItemChild(int type, int x, int y)
    {
        DiagramItem childItem = DiagramItem(type,/*m_context,*/this);
        childItem.xPosition=x;
        childItem.yPosition=y;
        items.add(childItem);
    }

    void addPort(QPointF center)
    {
        Port  newPort = Port();
        newPort->setItemParent(this);
        newPort->setAbsPoint(center);
        ports.add(newPort);
    }

    void addEquation()
    {
        Equation  newEquation = Equation(/*m_context,*/);
        equations.add(newEquation);
    }

    //void setContext(z3::context context) {m_context = context;}
    //z3::context context() const {return m_context;}

    void addResult(String variable, double result)
    {
        Result  newResult = Result(/*m_context,*/);
        newResult->setVarName(variable);
        newResult->setVarVal(result);
        results.add(newResult);
    }
}
