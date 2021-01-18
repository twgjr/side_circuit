import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0

Rectangle{
    id:blkRectId

    property int xPosition: model.xPos
    property int yPosition: model.yPos

    color: isSelected ? selectedColor : normalColor
    border.color: "black"
    border.width: 2
    radius: 5
    height: 100; width:100
    x: xPosition
    y: yPosition
    property int mouseBorder: 3*border.width
    property bool isSelected

    property color normalColor: "beige"
    property color selectedColor: "gold"

    property bool blockIsEditable: false

    Connections{
        target: flickableId
        function onBlockIsSelectedChanged(){
            var pos = flickableId.blockIsSelected.indexOf(model.index)
            if(pos === -1){
                isSelected = false
            } else {
                isSelected = true
            }
        }
    }

    Component.onCompleted: {
        isSelected = false
    }
    Component.onDestruction: {}



    MouseArea{
        id: blockMouseAreaId
        anchors.centerIn: parent
        height: parent.height - mouseBorder
        width: parent.width - mouseBorder
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        drag.target: parent
        drag.threshold: 0

        onDoubleClicked: {
            if(mouse.button & Qt.LeftButton){
                dataSource.downLevel(model.index)
            }
        }
        // single click and release
        onClicked: {
            if(mouse.button & Qt.RightButton){
                blockContextMenu.popup()
            }
            if(mouse.button & Qt.LeftButton){
                if(isSelected){
                    //unselect
                    var pos = flickableId.blockIsSelected.indexOf(model.index)
                    flickableId.blockIsSelected.splice(pos,1)
                    isSelected = false
                } else {
                    //select
                    flickableId.elementIsSelected.push(model.index)
                    isSelected = true
                }
            }
        }

        // single click and while being held
        onPressed: {}
        onReleased: {}
        onPositionChanged: {
            model.xPos = parent.x
            model.yPos = parent.y
            xPosition = model.xPos
            yPosition = model.yPos
            flickableId.maxFlickX = Math.max(dataSource.maxBlockX() + width*2, flickableId.width)
            flickableId.maxFlickY = Math.max(dataSource.maxBlockY() + height*2, flickableId.height)
        }

        Menu {
            id: blockContextMenu
            MenuItem {
                visible: !blockIsEditable
                text: "Add Port"
                onTriggered: {
                    //find position of port with lines crossing in an x
                    var posX = blockMouseAreaId.mouseX
                    var posY = blockMouseAreaId.mouseY
                    var dy = blockMouseAreaId.height
                    var dx = blockMouseAreaId.width
                    var xLineDown = dy/dx*posX
                    var xLineUp = dy-dy/dx*posX

                    var position = Math.min(blkRectId.width,blkRectId.height)/2
                    var side = 0
                    // top
                    if(posY<=xLineDown && posY<=xLineUp){
                        side = 0
                        position = posX
                    }
                    // bottom
                    if (posY>xLineDown && posY>xLineUp){
                        side = 1
                        position = posX
                    }
                    // left
                    if (posY>xLineDown && posY<xLineUp){
                        side = 2
                        position = posY
                    }
                    // right
                    if (posY<xLineDown && posY>xLineUp){
                        side = 3
                        position = posY
                    }
                    dataSource.addPort(0,model.index,side,position)
                }
            }//MenuItem
            MenuItem {
                enabled: !blockIsEditable
                text: "Down Level"
                onTriggered: {
                    dataSource.downLevel(model.index)
                }
            }
            MenuItem {
                enabled: !blockIsEditable
                text: "Delete"
                onTriggered: {
                    dataSource.deleteBlock(model.index)
                }
            }
            MenuItem {
                text: blockIsEditable ? "Finish Edit" : "Edit Block"
                onTriggered: {
                    blockIsEditable = !blockIsEditable
                }
            }
        } //Menu
    }

    Rectangle{
        id: bottomRightCornerId
        width: mouseBorder*2
        height: width
        anchors.horizontalCenter: parent.right
        anchors.verticalCenter: parent.bottom
        opacity: 0

        MouseArea{
            enabled: blockIsEditable
            anchors.fill: parent
            acceptedButtons: Qt.LeftButton
            hoverEnabled: true
            drag.target: parent
            drag.threshold: 0
            onEntered: cursorShape = Qt.SizeFDiagCursor

            onPositionChanged: {
                if(pressed){
                    blkRectId.height = Math.max(blkRectId.height+mouseY,blkRectId.border.width*4)
                    blkRectId.width = Math.max(blkRectId.width+mouseX,blkRectId.border.width*4)
                }
            }
        }
    }

    PortModel{
        id: blockPortModel
        proxyChildBlock: model.thisItem
        Component.onCompleted: {}
    }
    property int blockParentIndex: model.index
    Repeater{
        id : blockPortRepeater
        height: parent.height
        width: parent.width
        model : blockPortModel
        delegate: Port{ parentType: 0; parentIndex: blockParentIndex; portIsEditable: blockIsEditable }
    }

    ColumnLayout{
        anchors.fill: parent
        RowLayout{
            Text {
                Layout.fillWidth: true
                text : "Block ID:" + model.index
                horizontalAlignment: Text.AlignHCenter
            }
        }
    }
} //Rectangle

