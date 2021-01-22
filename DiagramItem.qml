import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0

Rectangle{
    id:itemRectId

    // properties tied to model
    property int itemParentIndex: model.index
    property int xPosition: model.xPos
    property int yPosition: model.yPos
    property int itemType: model.type
    property int itemRotation: model.rotation

    // QML properties specific to diagram item type (default block)
    property color normalColor: "beige"
    property color selectedColor: "gold"
    property color blockBorderColor: "black"
    property color elementBorderColor: "transparent"


    // properties standing alone in QML
    property int mouseBorder: 3*border.width
    property bool isSelected: false

    rotation: itemRotation
    color: isSelected ? selectedColor : normalColor
    border.width: 2
    radius: 5
    height: 100; width:100
    x: xPosition
    y: yPosition

    Component.onCompleted: {
        //isSelected = false
        switch(itemType){
        case DItemTypes.BlockItem:
            normalColor = "beige"
            selectedColor = "gold"
            border.color = blockBorderColor
            break;
        case DItemTypes.Resistor:
            normalColor = "transparent"
            selectedColor = "lightblue"
            border.color = elementBorderColor
            break;
        }
    }

    Connections{
        target: flickableId
        function onDItemIsSelectedChanged() {
            //itemIsEditable = false
            var pos = flickableId.dItemIsSelected.indexOf(model.index)
            if(pos === -1){
                isSelected = false

            } else {
                isSelected = true
            }
        }
    }
    Connections{
        target: rotateLeftButton
        function onClicked(){
            if(isSelected){
                itemRotation = (itemRotation - 45)%360
                model.rotation = itemRotation
            }
        }
    }
    Connections{
        target: rotateRightButton
        function onClicked(){
            if(isSelected){
                itemRotation = (itemRotation + 45)%360
                model.rotation = itemRotation
            }
        }
    }

    MouseArea{
        id: itemMouseAreaId
        anchors.centerIn: parent
        height: parent.height - mouseBorder
        width: parent.width - mouseBorder
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        drag.target: parent
        drag.threshold: 0

        onDoubleClicked: {
            if(mouse.button & Qt.LeftButton){
                if(itemType === DItemTypes.BlockItem){
                    dataSource.downLevel(model.index)
                }
            }
        }
        // single click and release
        onClicked: {
            if(mouse.button & Qt.RightButton){
                itemContextMenu.popup()
            }
            if(mouse.button & Qt.LeftButton){
                if(isSelected){
                    //unselect
                    var pos = flickableId.dItemIsSelected.indexOf(model.index)
                    flickableId.dItemIsSelected.splice(pos,1)
                    isSelected = false
                } else {
                    //select
                    flickableId.dItemIsSelected.push(model.index)
                    isSelected = true
                }
                console.log(model.thisItem)
                console.log(model.xPos)
                console.log(model.yPos)
                console.log(model.type)
                console.log(model.rotation)
            }
        }

        onPositionChanged: {
            model.xPos = parent.x
            model.yPos = parent.y
            xPosition = model.xPos
            yPosition = model.yPos
            flickableId.maxFlickX = Math.max(dataSource.maxItemX() + width*2, flickableId.width)
            flickableId.maxFlickY = Math.max(dataSource.maxItemY() + height*2, flickableId.height)
        }

        Menu {
            id: itemContextMenu
            MenuItem {
                text: "Add Port"
                onTriggered: {
                    //find position of port with lines crossing in an x
                    var posX = itemMouseAreaId.mouseX
                    var posY = itemMouseAreaId.mouseY
                    var dy = itemMouseAreaId.height
                    var dx = itemMouseAreaId.width
                    var xLineDown = dy/dx*posX
                    var xLineUp = dy-dy/dx*posX

                    var position = Math.min(itemRectId.width,itemRectId.height)/2
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
                    dataSource.addPort(model.index,side,position)
                }
            }//MenuItem
            MenuItem {
                visible: itemType===DItemTypes.BlockItem ? true : false
                height: itemType===DItemTypes.BlockItem ? implicitHeight : 0
                text: "Down Level"
                onTriggered: {
                    dataSource.downLevel(model.index)
                }
            }
            MenuItem {
                text: "Delete"
                onTriggered: {
                    dataSource.deleteDiagramItem(model.index)
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
            anchors.fill: parent
            acceptedButtons: Qt.LeftButton
            hoverEnabled: true
            drag.target: parent
            drag.threshold: 0
            onEntered: cursorShape = Qt.SizeFDiagCursor

            onPositionChanged: {
                if(pressed){
                    itemRectId.height = Math.max(itemRectId.height+mouseY,itemRectId.border.width*4)
                    itemRectId.width = Math.max(itemRectId.width+mouseX,itemRectId.border.width*4)
                }
            }
        }
    }

    PortModel{
        id: portModel
        parentItem: model.thisItem
    }

    Repeater{
        id : portRepeater
        height: parent.height
        width: parent.width
        model : portModel
        delegate: Port{ parentIndex: itemParentIndex }
    }

    ColumnLayout{
        anchors.fill: parent
        RowLayout{
            Text {
                Layout.fillWidth: true
                text : "Item ID:" + model.index
                horizontalAlignment: Text.AlignHCenter
            }
        }
    }
} //Rectangle

