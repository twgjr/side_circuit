import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0
import "portScript.js" as PortScript

Rectangle {
    id: portId

    // QML "constructor" properties
    property int parentIndex

    // properties tied to model
    property int proxyPortIndex: model.index
//    property int sideNum: model.side
//    property int positionNum: model.position
    property string nameText: model.name
    property point absPoint: model.absPoint

//    // properties standing alone in QML
//    property int leftBound: 0
//    property int rightBound: parent.width
//    property int topBound: 0
//    property int bottomBound: parent.height

    width: 10
    height: width
    radius: width/2
    border.color: "black"
    border.width: 2

    signal portAbsPositionChanged()

    Connections{
        target: itemMouseAreaId
        function onPositionChanged() {
            updatePortAbsPosition()
            portAbsPositionChanged()
        }
    }

    Connections{
        target: itemRectId
        function onRotationChanged() {
            updatePortAbsPosition()
            portAbsPositionChanged()
        }
    }

    function updatePortAbsPosition(){
        var itemAngle = itemRectId.rotation/180*Math.PI
        var itemCenter = [itemRectId.width/2,itemRectId.height/2]
        var portBase = [portId.x,portId.y]
        var portCenterDelta = [portId.width/2,portId.height/2]
        var portCenter = [
                    portBase[0]+portCenterDelta[0],
                    portBase[1]+portCenterDelta[1]]
        var portCenterFromItemCenter = [
                    portCenter[0]-itemCenter[0],
                    portCenter[1]-itemCenter[1]]
        var portItemDist = Math.sqrt(
                    Math.pow(Math.abs(portCenterFromItemCenter[0]),2)+
                    Math.pow(Math.abs(portCenterFromItemCenter[1]),2))
        var portBaseAngle = Math.atan2(
                    portCenterFromItemCenter[0],
                    -portCenterFromItemCenter[1])
        var portRotatedCenter = [
                    portItemDist*Math.sin(portBaseAngle+itemAngle),
                    -portItemDist*Math.cos(portBaseAngle+itemAngle)]
        var portRotatedBase = [
                    portRotatedCenter[0]+itemCenter[0],
                    portRotatedCenter[1]+itemCenter[1]]

        // absolute point within the diagram mouse area
        var absRotatedPt = [
                    itemRectId.x+portRotatedBase[0],
                    itemRectId.y+portRotatedBase[1]]
        absPoint = Qt.point(absRotatedPt[0],absRotatedPt[1])
        model.absPoint = absPoint
        console.log("Port updated: "+absPoint)
        dataSource.resetConnectedLinkstoPort(model.thisPort)
        dataSource.resetLinkstoPort(model.thisPort)
    }

    Component.onCompleted: {
        portId.x = absPoint.x-itemRectId.x
        portId.y = absPoint.y-itemRectId.y
        updatePortAbsPosition()
    }

    DropArea{
        anchors.fill: parent
        anchors.margins: -parent.radius/2

        onDropped: {
            console.log("dropped")
            model.connectionIsValid = true
            if(model.connectionIsValid){
                dataSource.endLinkFromPort(model.thisPort)
                updatePortAbsPosition()
            }
        }

        onContainsDragChanged: {
            if(containsDrag){
                Drag.keys = [ "dropKey" ] // check on drag hover over drop area
                console.log("CHECK PORT COMPATIBILITY. SHOW SPECIAL CURSOR")
            }
        }
    }

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
            if(mouse.button & Qt.LeftButton){
                console.log(model.index)
                console.log(model.name)
                console.log(model.thisPort)
            }
        }
        onPositionChanged: {
            var mousePosX = mouseX-parent.radius+parent.x
            var mousePosY = mouseY-parent.radius+parent.y
            updatePortAbsPosition()

        }//onPositionChanged
    }//MouseArea
    Label{
        text: nameText
        Component.onCompleted: {
            x = parent.width
            y = parent.height
        }
        MouseArea{
            anchors.fill: parent
            drag.target: parent
            drag.threshold: 0
        }
    }
    Menu {
        id: portContextMenu
        MenuItem {
            text: "Start Link"
            onTriggered: {
                dataSource.startLink(parentIndex,model.index)
            }
        }
        MenuItem {
            text: "Delete Port"
            onTriggered: {
                dataSource.deletePort(parentIndex,model.index)
            }
        }
    } //Menu

    LinkModel{
        id: linkModel
        proxyPort: model.thisPort
    }

    Repeater{
        id : portRepeater
        height: parent.height
        width: parent.width
        model : linkModel
        delegate: Link{}
    }
}
