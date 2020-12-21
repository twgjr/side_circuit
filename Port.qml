import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0
import "portScript.js" as PortScript

Rectangle {
    id: portId

    property int sideNum: model.side
    property int positionNum: model.position
    property string nameText: model.name

    width: 10
    height: width
    radius: width/2
    border.color: "black"
    border.width: 2

    property int offSet: width/2
    property int leftBound: -offSet
    property int rightBound: blkRectId.width - offSet
    property int topBound: -offSet
    property int bottomBound: blkRectId.height - offSet

    Component.onCompleted: {
        switch (sideNum){
        case 0://top
            portMouseArea.drag.axis = Drag.XAxis
            portId.x = positionNum
            portId.anchors.verticalCenter = blkRectId.top
            break;
        case 1://bottom
            portMouseArea.drag.axis = Drag.XAxis
            portId.x = positionNum
            portId.anchors.verticalCenter = blkRectId.bottom
            break;
        case 2://left
            portMouseArea.drag.axis = Drag.YAxis
            portId.y = positionNum
            portId.anchors.horizontalCenter = blkRectId.left
            break;
        case 3://right
            portMouseArea.drag.axis = Drag.YAxis
            portId.y = positionNum
            portId.anchors.horizontalCenter = blkRectId.right
            break;
        }
    }

    MouseArea {
        id: portMouseArea
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        anchors.fill: parent
        drag.target: parent
        drag.threshold: 0

        drag.minimumX: leftBound+offSet
        drag.maximumX: rightBound-offSet
        drag.minimumY: topBound+offSet
        drag.maximumY: bottomBound-offSet

        onClicked: {
            if(mouse.button & Qt.RightButton){
                portContextMenu.popup()
            }
        }
    }
    Label{
        text: nameText
        anchors.top: parent.bottom
        anchors.left: parent.right
    }
    Menu {
        id: portContextMenu
        MenuItem {
            text: "Delete"
            onTriggered: {
                proxyPortsId.deletePort(model.index)
            }
        }
    } //Menu
}
