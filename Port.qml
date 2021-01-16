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

    property int sideNum: model.side
    property int positionNum: model.position
    property string nameText: model.name
    property int parentType //0:block,1:element
    property int parentIndex
    property bool portIsEditable: false

    width: 10
    height: width
    radius: width/2
    border.color: "black"
    border.width: 2

    property int leftBound: 0
    property int rightBound: parent.width
    property int topBound: 0
    property int bottomBound: parent.height

    function setSide(sideType){
        anchors.horizontalCenter = undefined
        anchors.verticalCenter = undefined

        switch(sideType){
        case "top":
            sideNum = 0
            anchors.verticalCenter = parent.top
            positionNum = portMouseArea.mouseX-radius
            break;
        case "bottom":
            sideNum = 1
            anchors.verticalCenter = parent.bottom
            positionNum = portMouseArea.mouseX-radius
            break;
        case "left":
            sideNum = 2
            anchors.horizontalCenter = parent.left
            positionNum = portMouseArea.mouseY-radius
            break;
        case "right":
            sideNum = 3
            anchors.horizontalCenter = parent.right
            positionNum = portMouseArea.mouseY-radius
            break;
        }

        model.side = sideNum
        model.position = positionNum
    }

    Component.onCompleted: {
        switch (sideNum){
        case 0://top
            x = positionNum
            anchors.verticalCenter = parent.top
            break;
        case 1://bottom
            x = positionNum
            anchors.verticalCenter = parent.bottom
            break;
        case 2://left
            y = positionNum
            anchors.horizontalCenter = parent.left
            break;
        case 3://right
            y = positionNum
            anchors.horizontalCenter = parent.right
            break;
        }
    }

    //property string dropKey: "dropKey"
    DropArea{
        anchors.fill: parent
        Drag.keys: [ "dropKey" ]

        onDropped: {
            console.log("dropped")
        }
        onEntered: {
            portId.color = "green"
            console.log("entered")
        }
        onExited: {
            portId.color = "orange"
            console.log("exited")
        }
    }



    MouseArea {
        id: portMouseArea
        enabled: portIsEditable
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        anchors.fill: parent
        drag.target: parent
        drag.threshold: 0

        drag.minimumX: parent.leftBound-parent.radius
        drag.maximumX: parent.rightBound-parent.radius
        drag.minimumY: parent.topBound-parent.radius
        drag.maximumY: parent.bottomBound-parent.radius



        onClicked: {
            if(mouse.button & Qt.RightButton){
                portContextMenu.popup()
            }
        }
        onPositionChanged: {
            var mousePosX = mouseX-parent.radius+parent.x
            var mousePosY = mouseY-parent.radius+parent.y

            switch(sideNum){
            case 0:// top
                if( mousePosY > parent.topBound){
                    if( mousePosX < parent.leftBound){
                        setSide("left")
                    }
                    if( mousePosX > parent.rightBound ){
                        setSide("right")
                    }
                }
                if( mousePosY > parent.bottomBound ){
                    if( parent.leftBound < mousePosX < parent.rightBound){
                        setSide("bottom")
                    }
                }
                break;
            case 1:// bottom
                if( mousePosY < parent.bottomBound ){
                    if( mousePosX < parent.leftBound){
                        setSide("left")
                    }
                    if( mousePosX > parent.rightBound ){
                        setSide("right")
                    }
                }
                if( mousePosY < parent.topBound ){
                    if( parent.leftBound < mousePosX < parent.rightBound){
                        setSide("top")
                    }
                }
                break;
            case 2:// left
                if( mousePosX > parent.leftBound ){
                    if( mousePosY < parent.topBound ){
                        setSide("top")
                    }
                    if( mousePosY > parent.bottomBound ){
                        setSide("bottom")
                    }
                }
                if( mousePosX > parent.rightBound ){
                    if( parent.topBound < mousePosY < parent.bottomBound){
                        setSide("right")
                    }
                }
                break;
            case 3:// right
                if( mousePosX < parent.rightBound ){
                    if( mousePosY < parent.topBound ){
                        setSide("top")
                    }
                    if( mousePosY > parent.bottomBound ){
                        setSide("bottom")
                    }
                }
                if( mousePosX < parent.leftBound ){
                    if( parent.topBound < mousePosY < parent.bottomBound){
                        setSide("left")
                    }
                }
                break;
            default:
                break;
            }//switch
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
            visible: !portIsEditable
            text: "Start Link"
            onTriggered: {
                dataSource.startLink(parentType,parentIndex,model.index)
            }
        }
        MenuItem {
            visible: portIsEditable
            text: "Delete Port"
            onTriggered: {
                //dataSource.deletePort(proxyParentIndex,model.index)
            }
        }
    } //Menu

    LinkModel{
        id: linkModel
        proxyPort: model.thisPort
        Component.onCompleted: {
        }
    }
    property int proxyPortIndex: model.index
    Repeater{
        id : portRepeater
        height: parent.height
        width: parent.width
        model : linkModel
        delegate: Link{}
    }
}
