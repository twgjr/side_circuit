import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import QtQuick.Shapes 1.15
import Qt.labs.qmlmodels 1.0
import com.company.models 1.0

MouseArea{
    acceptedButtons: Qt.LeftButton | Qt.RightButton
    drag.target: parent
    drag.threshold: 0
    property int posX
    property int posY

    onDoubleClicked: {
        if(mouse.button & Qt.LeftButton){
            flickableId.leveltext = dataSource.distanceFromRoot()+1
            dataSource.downLevel(model.index)
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
        model.xPos = blkRectId.x
        model.yPos = blkRectId.y
        xPosition = model.xPos
        yPosition = model.yPos
        flickableId.maxFlickX = Math.max(dataSource.maxBlockX() + width*2, flickableId.width)
        flickableId.maxFlickY = Math.max(dataSource.maxBlockY() + height*2, flickableId.height)
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
            text: "Down Level"
            onTriggered: {
                flickableId.leveltext = dataSource.distanceFromRoot()+1
                dataSource.downLevel(model.index)
            }
        }
        MenuItem {
            text: "Delete"
            onTriggered: {
                dataSource.deleteBlock(model.index)
            }
        }
    } //Menu
}
