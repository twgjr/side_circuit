import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0

Item{
    id: linkItemRoot
    height:implicitHeight
    width:implicitWidth

    Shape {
        id:shapeRoot
        ShapePath {
            id:shapeRootPath
            strokeWidth: 2
            strokeColor: "red"
            strokeStyle: ShapePath.DashLine
            dashPattern: [ 1, 5 ]
            startX: portId.width/2 ; startY: portId.height/2
            PathLine {
                id: segment
                x: endRect.x+endRect.radius
                y: endRect.y+endRect.radius
            }
        }
    }//Shape

    Rectangle{
        id: endRect

        property var endRectPoint: model.lastPoint
        property real endRectX: endRectPoint.x
        property real endRectY: endRectPoint.y

        width: 10
        height: width
        radius: width/2
        border.color: "black"
        color: "transparent"

        Drag.active: linkMouseArea.drag.active
        Drag.keys: [ "dropKey" ]
        Drag.hotSpot.x: width/2
        Drag.hotSpot.y: height/2

        states: [
            State {
                name: "connected"; when: model.portConnected
                PropertyChanges {
                    target: linkMouseArea
                    drag.target: undefined
                    x: endRect.endRectX-itemRectId.x-portId.x
                    y: endRect.endRectY-itemRectId.y-portId.y
                }
                PropertyChanges {
                    target: shapeRootPath
                    strokeColor: "black"
                    strokeStyle: ShapePath.SolidLine
                }
            }
        ]

        Component.onCompleted:{
            if(model.portConnected){
                x = endRectX-itemRectId.x-portId.x
                y = endRectY-itemRectId.y-portId.y
            } else {
                var extraX = 0
                var extraY = 0
                var extra = 10
                switch(sideNum){
                case 0: extraY = -extra; break;
                case 1: extraY = extra; break;
                case 2: extraX = -extra; break;
                case 3: extraX = extra; break;
                }
                x = extraX
                y = extraY
            }
        }

        MouseArea {
            id: linkMouseArea
            acceptedButtons: Qt.LeftButton | Qt.RightButton
            anchors.fill: parent
            drag.target: parent
            drag.threshold: 0
            drag.smoothed: false

            onPositionChanged: {
                if(!model.portConnected){
                    var linkAbsX = itemRectId.x+portId.x+endRect.x
                    var linkAbsY = itemRectId.y+portId.y+endRect.y
                    model.lastPoint = Qt.point(linkAbsX,linkAbsY)
                }
            }

            onReleased: {
                //drop handler
                if(endRect.Drag.target !== null){
                    console.log("drag target: "+endRect.Drag.target)
                    dataSource.endLinkFromLink(model.thisLink)
                } else {
                    console.log("drag target was null")
                }
                parent.Drag.drop()
            }

            onClicked: {
                if(mouse.button & Qt.RightButton){
                    portContextMenu.popup()

                }
                if(mouse.button & Qt.LeftButton){
                    //add new segment and set new segment to follow mouse pointer
                }
            }

            onPressed: {
                endRect.state = ""
                dataSource.disconnectPortfromLink(model.thisLink)
            }


        }//MouseArea
    }//Rectangle

    Menu {
        id: portContextMenu
        MenuItem {
            text: "Delete Link"
            onTriggered: {
                dataSource.deleteLink(parentIndex,proxyPortIndex,model.index)
            }
        }
    } //Menu
}//Item
