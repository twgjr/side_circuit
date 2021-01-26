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
    property int sideNum: model.side
    property int positionNum: model.position
    property string nameText: model.name
    property point absPoint: model.absPoint

    // properties standing alone in QML
    property int leftBound: 0
    property int rightBound: parent.width
    property int topBound: 0
    property int bottomBound: parent.height

    width: 10
    height: width
    radius: width/2
    border.color: "black"
    border.width: 2

    Connections{
        target: itemMouseAreaId
        function onPositionChanged() {
            updateLinkPosition()
        }
    }

    Connections{
        target: itemRectId
        function onRotationChanged() {
            updateLinkPosition()
        }
    }

    function updateLinkPosition(){
        var rotateAngle = itemRectId.rotation/180*Math.PI
        var centerPt = [itemRectId.width/2,itemRectId.height/2]
        var basePt_O = [portId.x,portId.y]
        var basePt_C = [basePt_O[0]-centerPt[0],basePt_O[1]-centerPt[1]]
        var rd = Math.sqrt(Math.pow(Math.abs(basePt_C[0]),2)+Math.pow(Math.abs(basePt_C[1]),2))
        console.log("rd: "+rd)
        var fixedAngle = [Math.asin(basePt_C[0]/rd),Math.acos(-basePt_C[1]/rd)]
        var rotatedPt_C = [rd*Math.sin(fixedAngle[0]-rotateAngle),-rd*Math.cos(fixedAngle[1]-rotateAngle)]
        var rotatedPt_O = [rotatedPt_C[0] + centerPt[0],rotatedPt_C[1]+centerPt[1]]
        var absRotatedPt = [itemRectId.x+rotatedPt_O[0],itemRectId.y+rotatedPt_O[1]]
        console.log("rotation: "+itemRectId.rotation)
        console.log("basePt_O: "+basePt_O)
        console.log("basePt_C: "+basePt_C)
        console.log("rotatedPt_C: "+rotatedPt_C)
        console.log("rotatedPt_O: "+rotatedPt_O)
        console.log("absRotatedPt: "+absRotatedPt)
        model.absPoint = Qt.point(absRotatedPt[0],absRotatedPt[1])
        dataSource.resetConnectedLinkstoPort(model.thisPort)
        dataSource.resetLinkstoPort(model.thisPort)
    }

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
        updateLinkPosition()
    }

    //property string dropKey: "dropKey"
    DropArea{
        anchors.fill: parent
        anchors.margins: -parent.radius/2

        //Drag.keys: [ "dropKey" ]

        onDropped: {
            console.log("dropped")
            dataSource.endLinkFromPort(model.thisPort)
        }
//        onEntered: {
//            portId.color = "green"
//            console.log("entered")
//        }
//        onExited: {
//            portId.color = "orange"
//            console.log("exited")
//        }
        onContainsDragChanged: {
            if(containsDrag){
                console.log("check the link for compatibility")
                Drag.keys = [ "dropKey" ]
            }
        }
    }

    MouseArea {
        id: portMouseArea
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        anchors.fill: parent
        drag.target: parent
        drag.threshold: 0
        //hoverEnabled: true

        drag.minimumX: parent.leftBound-parent.radius
        drag.maximumX: parent.rightBound-parent.radius
        drag.minimumY: parent.topBound-parent.radius
        drag.maximumY: parent.bottomBound-parent.radius

        onClicked: {
            if(mouse.button & Qt.RightButton){
                portContextMenu.popup()
            }
            if(mouse.button & Qt.LeftButton){
                console.log(model.index)
                console.log(model.side)
                console.log(model.position)
                console.log(model.name)
                console.log(model.thisPort)
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

            updateLinkPosition()

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
