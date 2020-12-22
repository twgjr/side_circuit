import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import QtQuick.Shapes 1.15
import com.company.models 1.0

Rectangle{
    width:10
    height: width
    color: "red"
    anchors.left: portId.right

    MouseArea {
        id: portMouseArea
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        anchors.fill: parent
        drag.target: parent
        drag.threshold: 0

        onClicked: {
            if(mouse.button & Qt.RightButton){
                portContextMenu.popup()
            }
        }
    }

    Menu {
        id: portContextMenu
        MenuItem {
            text: "Delete Link"
            onTriggered: {
                //proxyBlockId.deletePort(model.index)
                dsPortId.deleteLink(model.index)
            }
        }
    } //Menu
}

//Shape {
//    width: implicitHeight
//    height: implicitWidth
//    //anchors.centerIn: parent
//    ShapePath {
//        strokeWidth: 4
//        strokeColor: "red"
//        strokeStyle: ShapePath.DashLine
//        dashPattern: [ 1, 4 ]
//        startX: portId.x ; startY: portId.y
//        PathLine { x: 20; y: 130 }
//    }

//    MouseArea {
//        id: portMouseArea
//        acceptedButtons: Qt.LeftButton | Qt.RightButton
//        anchors.fill: parent
//        drag.target: parent
//        drag.threshold: 0

//        onClicked: {
//            if(mouse.button & Qt.RightButton){
//                portContextMenu.popup()
//            }
//        }
//    }
//    Menu {
//        id: portContextMenu
//        MenuItem {
//            text: "Delete Link"
//            onTriggered: {
//                dsLinksId.deleteLink(model.index)
//            }
//        }
//    } //Menu
//}
