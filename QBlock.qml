import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0

Rectangle{
    id: blkRectId
    property int idText: model.id
    property string descriptionText: model.description
    property int xPosition: model.blockXPosition
    property int yPosition: model.blockYPosition
    property string equationText: model.equationString
    property bool selected: false


    color: "beige"
    border.color: "black"
    border.width: 2
    radius: Math.min(height,width)*0.1
    height: 100; width:100
    x: xPosition
    y: yPosition

    transform: Scale{id:blockScaleId}

    Component.onCompleted: {}
    Component.onDestruction: {}

    property int mouseBorder: 2*border.width
    BlockMouseArea{
        id: blockMouseAreaId
        anchors.centerIn: parent
        height: parent.height - mouseBorder
        width: parent.width - mouseBorder
    }
    BlockCornerMouseArea {
        id: bottomRightCornerId
        height: mouseBorder
        width: mouseBorder
        anchors.left: blockMouseAreaId.right
        anchors.top: blockMouseAreaId.bottom
    }

    Menu {
        id: blockContextMenu
        MenuItem {
            text: "Add Port"
            onTriggered: {
                //find position of port with lines crossing in an x
                var posX=blockMouseArea.mouseX
                var posY=blockMouseArea.mouseY
                var dy = blockMouseArea.height
                var dx = blockMouseArea.width
                var xLine1 = dy/dx*posX
                var xLine2 = -dy/dx*posX+posY

                console.log("index: "+model.index)
                console.log("x lines: "+xLine1+" x "+xLine2)
                console.log("pointer: "+posX+" x "+posY)
                console.log("area: "+dx+" x "+dy)

                // top
                if(posY>xLine1 && posY>xLine2){myBlockModel.addPort(model.index,0,posX)}
                // bottom
                if (posY<xLine1 && posY<xLine2){myBlockModel.addPort(model.index,1,posX)}
                // left
                if (posY<xLine1 && posY>xLine2){myBlockModel.addPort(model.index,2,posY)}
                // right
                if (posY>xLine1 && posY<xLine2){myBlockModel.addPort(model.index,3,posY)}
            }
        }//MenuItem
        MenuItem {
            text: "Down Level"
            onTriggered: myBlockModel.downLevel(model.index)
        }
        MenuItem {
            text: "Delete"
            onTriggered: myBlockModel.deleteBlock(model.index)
        }
        MenuItem {
            text: "Set Block Color"
            onTriggered: blockColorDialog.open()
        }
        ColorDialog {
            id: blockColorDialog
            currentColor: blkRectId.color
            title: "Please choose a block color"
            onAccepted: {
                console.log("You chose: " + color)
                blkRectId.color = color
                close()
            }
            onRejected: {
                console.log("Canceled")
                close()
            }
        }
    } //Menu

    ColumnLayout{
        anchors.fill: parent
        RowLayout{
            Text {
                Layout.fillWidth: true
                text : "Block ID:" + idText
                horizontalAlignment: Text.AlignHCenter
            }
        }
    }
} //Rectangle