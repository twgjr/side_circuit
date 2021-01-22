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
            strokeWidth: 4
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
        width: 10
        height: width
        radius: width/2
        border.color: "black"
        color: "transparent"

        Drag.active: linkMouseArea.drag.active
        Drag.keys: [ "dropKey" ]
        Drag.hotSpot.x: width/2
        Drag.hotSpot.y: height/2

        Component.onCompleted: {
            x = diagramMouseArea.mouseX-itemRectId.x-portId.x
            y = diagramMouseArea.mouseY-itemRectId.y-portId.y
        }

        MouseArea {
            id: linkMouseArea
            acceptedButtons: Qt.LeftButton | Qt.RightButton
            anchors.fill: parent
            drag.target: parent
            drag.threshold: 0
            drag.smoothed: false

            onReleased: {
                //drop handler
                if(endRect.Drag.target !== null){
                    console.log("drag target: "+endRect.Drag.target)
                    endRect.parent = endRect.Drag.target
                    endRect.anchors.verticalCenter = endRect.Drag.target.verticalCenter
                    endRect.anchors.horizontalCenter = endRect.Drag.target.horizontalCenter
                } else {
                    console.log("drag target was null")
                    endRect.parent = linkItemRoot
                    endRect.anchors.verticalCenter = undefined
                    endRect.anchors.horizontalCenter = undefined
                }
                //parent.Drag.drop()
            }

            onClicked: {
                if(mouse.button & Qt.RightButton){
                    portContextMenu.popup()
                    //cancel the link drag
                }
                if(mouse.button & Qt.LeftButton){
                    //add new segment and set new segment to follow mouse pointer
                }
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
