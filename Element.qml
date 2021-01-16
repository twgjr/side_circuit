import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0
import "elementScript.js" as ElementScript

Rectangle{
    id:elementRectId

    property int xPosition: model.xPos
    property int yPosition: model.yPos
    property int elementType: model.type
    property int elementRotation: model.rotation

    property bool isSelected

    property color normalColor: "transparent"
    property color selectedColor: "lightblue"

    property bool elementIsEditable: false

    Connections{
        target: flickableId
        function onElementIsSelectedChanged(){
            var pos = flickableId.elementIsSelected.indexOf(model.index)
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
                elementRotation = (elementRotation - 45)%360
                model.rotation = elementRotation
            }
        }
    }

    Connections{
        target: rotateRightButton
        function onClicked(){
            if(isSelected){
                elementRotation = (elementRotation + 45)%360
                model.rotation = elementRotation
            }
        }
    }

    rotation: elementRotation
    color: isSelected ? selectedColor : normalColor
    radius: 5
    height: 100; width:100
    x: xPosition
    y: yPosition

    Component.onCompleted: {
        var selectType
        switch(elementType){
        case 0: selectType="GenericIC.qml"; break;
        case 1: selectType="Resistor.qml"; break;
        case 2: selectType="Capacitor.qml"; break;
        case 3: selectType="Inductor.qml"; break;
        case 4: selectType="SSSW.qml"; break;
        case 5: selectType="Diode.QML"; break;
        case 6: selectType="LED.qml"; break;
        case 7: selectType="Battery.qml"; break;
        case 8: selectType="Connector.qml"; break;
        }
        ElementScript.createSpriteObjects(selectType)
        isSelected = false
    }

    property int mouseBorder: 3*border.width

    MouseArea{
        id: elementMouseAreaId
        anchors.centerIn: parent
        height: parent.height - mouseBorder
        width: parent.width - mouseBorder
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        drag.target: parent
        drag.threshold: 0

        onDoubleClicked: {
            if(mouse.button & Qt.LeftButton){
                //open advanced element properties and options dialog
            }
        }
        // single click and release
        onClicked: {
            if(mouse.button & Qt.RightButton){
                elementContextMenu.popup()
            }
            if(mouse.button & Qt.LeftButton){
                if(isSelected){
                    //unselect
                    var pos = flickableId.elementIsSelected.indexOf(model.index)
                    flickableId.elementIsSelected.splice(pos,1)
                    isSelected = false
                } else {
                    //select
                    flickableId.elementIsSelected.push(model.index)
                    isSelected = true
                }
            }
        }

        onPositionChanged: {
            model.xPos = parent.x
            model.yPos = parent.y
            xPosition = model.xPos
            yPosition = model.yPos
            flickableId.maxFlickX = Math.max(dataSource.maxBlockX() + width*2, flickableId.width)
            flickableId.maxFlickY = Math.max(dataSource.maxBlockY() + height*2, flickableId.height)
        }

        Menu {
            id: elementContextMenu
//            MenuItem {
//                text: "Add Port"
//                onTriggered: {
//                    //find position of port with lines crossing in an x
//                    var posX = elementMouseAreaId.mouseX
//                    var posY = elementMouseAreaId.mouseY
//                    var dy = elementMouseAreaId.height
//                    var dx = elementMouseAreaId.width
//                    var xLineDown = dy/dx*posX
//                    var xLineUp = dy-dy/dx*posX

//                    var position = Math.min(elementRectId.width,elementRectId.height)/2
//                    var side = 0
//                    // top
//                    if(posY<=xLineDown && posY<=xLineUp){
//                        side = 0
//                        position = posX
//                    }
//                    // bottom
//                    if (posY>xLineDown && posY>xLineUp){
//                        side = 1
//                        position = posX
//                    }
//                    // left
//                    if (posY>xLineDown && posY<xLineUp){
//                        side = 2
//                        position = posY
//                    }
//                    // right
//                    if (posY<xLineDown && posY>xLineUp){
//                        side = 3
//                        position = posY
//                    }
//                    dataSource.addPort(1,model.index,side,position)
//                }
//            }//MenuItem

            MenuItem {
                enabled: !elementIsEditable
                text: "Delete"
                onTriggered: {
                    dataSource.deleteElement(model.index)
                }
            }
            MenuItem {
                text: elementIsEditable ? "Finish Edit" : "Edit Element"
                onTriggered: {
                    elementIsEditable = !elementIsEditable
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
                    elementRectId.height = Math.max(elementRectId.height+mouseY,elementRectId.border.width*4)
                    elementRectId.width = Math.max(elementRectId.width+mouseX,elementRectId.border.width*4)
                }
            }
        }
    }

    PortModel{
        id: elementPortModel
        proxyElement: model.thisItem
    }
    property int elementParentIndex: model.index
    Repeater{
        id : elementPortRepeater
        height: parent.height
        width: parent.width
        model : elementPortModel
        delegate: Port{ parentType: 1; parentIndex: elementParentIndex; portIsEditable: elementIsEditable }
    }

    ColumnLayout{
        anchors.fill: parent
        Text {
            Layout.fillWidth: true
            text : "Element ID:" + model.index
            horizontalAlignment: Text.AlignHCenter
        }
        Text {
            Layout.fillWidth: true
            text : "Element Type:" + elementType
            horizontalAlignment: Text.AlignHCenter
        }
    }

    Dialog{
        id:elementDialog
        RowLayout{
            Label{
                text: "Element Type"
            }
        }
        onAccepted: {
            equationString = equationText.text
            model.equationString = equationString
            close()
        }
    }//Dialog
} //Rectangle

