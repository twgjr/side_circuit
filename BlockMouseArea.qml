import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0

MouseArea{
    acceptedButtons: Qt.LeftButton | Qt.RightButton
    drag.target: parent

    onDoubleClicked: {
        if(mouse.button & Qt.LeftButton){
            flickableId.leveltext = myBlockModel.distanceFromRoot()+1
            myBlockModel.downLevel(model.index)
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
        flickableId.maxFlickX = Math.max(myBlockModel.maxBlockX() + width*2, flickableId.width)
        flickableId.maxFlickY = Math.max(myBlockModel.maxBlockY() + height*2, flickableId.height)
    }
}
