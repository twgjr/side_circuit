import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0
//import "portCreation.js" as PortScript

MouseArea{
    acceptedButtons: Qt.LeftButton | Qt.RightButton
    drag.target: parent
    drag.threshold: 0
    property int posX
    property int posY

    onDoubleClicked: {
        if(mouse.button & Qt.LeftButton){
            flickableId.leveltext = blockModel.distanceFromRoot()+1
            blockModel.downLevel(model.index)
        }
    }
    // single click and release
    onClicked: {
        if(mouse.button & Qt.RightButton){
            blockContextMenu.popup()
        }
    }
    // single click and while being held
    onPressed: {}
    onReleased: {}
    onPositionChanged: {
        model.blockXPosition = blkRectId.x
        model.blockYPosition = blkRectId.y
        xPosition = model.blockXPosition
        yPosition = model.blockYPosition
        flickableId.maxFlickX = Math.max(blockModel.maxBlockX() + width*2, flickableId.width)
        flickableId.maxFlickY = Math.max(blockModel.maxBlockY() + height*2, flickableId.height)
    }

    Menu {
        id: blockContextMenu
        MenuItem {
            text: "Add Port"
            onTriggered: {
                //find position of port with lines crossing in an x
                blockMouseAreaId.posX = blockMouseAreaId.mouseX
                blockMouseAreaId.posY = blockMouseAreaId.mouseY
                var dy = blockMouseAreaId.height
                var dx = blockMouseAreaId.width
                var xLineDown = dy/dx*posX
                var xLineUp = dy-dy/dx*posX

                var position = Math.min(blkRectId.width,blkRectId.height)/2
                var side = 0
                // top
                if(posY<=xLineDown && posY<=xLineUp){
                    //console.log("port on top")
                    side = 0
                    position = posX
                }
                // bottom
                if (posY>xLineDown && posY>xLineUp){
                    //console.log("port on bottom")
                    side = 1
                    position = posX
                }
                // left
                if (posY>xLineDown && posY<xLineUp){
                    //console.log("port on left")
                    side = 2
                    position = posY
                }
                // right
                if (posY<xLineDown && posY>xLineUp){
                    //console.log("port on right")
                    side = 3
                    position = posY
                }
                //PortScript.createPortObjects(side,position)
                portModel.addPort(side,position)
            }
        }//MenuItem
        MenuItem {
            text: "Down Level"
            onTriggered: {
                flickableId.leveltext = blockModel.distanceFromRoot()+1
                blockModel.downLevel(model.index)
            }
        }
        MenuItem {
            text: "Delete"
            onTriggered: blockModel.deleteBlock(model.index)
        }
    } //Menu
}
