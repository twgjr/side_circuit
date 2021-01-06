import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import QtQuick.Shapes 1.15
import Qt.labs.qmlmodels 1.0
import com.company.models 1.0

Shape {
    anchors.centerIn: parent
    ShapePath {
        strokeWidth: 4
        strokeColor: "red"
        strokeStyle: ShapePath.DashLine
        dashPattern: [ 1, 4 ]
        startX: 0 ; startY: 0
        PathLine {
            id: segment
            x: endRect.x+endRect.radius
            y: endRect.y+endRect.radius
        }
    }
    Rectangle{
        id: endRect
        width: 10
        height: width
        radius: width/2
        border.color: "black"
        color: "transparent"

        Drag.active: linkMouseArea.drag.active
        Drag.keys: [ "dropKey" ]

        //property string dropKey: "dropKey"
        MouseArea {
            id: linkMouseArea
            acceptedButtons: Qt.LeftButton | Qt.RightButton
            anchors.fill: parent
            drag.target: parent
            drag.threshold: 0

            onReleased: {
                parent.Drag.drop()
            }

            onClicked: {
                if(mouse.button & Qt.RightButton){
                    portContextMenu.popup()
                }
                if(mouse.button & Qt.LeftButton){
                    console.log("add new segment and set new segment to follow mouse pointer")
                }
            }
            onPositionChanged: {}
        }//MouseArea
    }//Rectangle

    Menu {
        id: portContextMenu
        MenuItem {
            text: "Delete Link"
            onTriggered: {
                dataSource.deleteLink(proxyBlockIndex,proxyPortIndex,model.index)
            }
        }
    } //Menu
}//Shape
