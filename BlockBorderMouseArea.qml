import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0

MouseArea{
    acceptedButtons: Qt.LeftButton

    property string borderRegion
    property int oldMouseX
    property int oldMouseY
    property int mouseDX
    property int mouseDY
    property bool leftMarginClicked: false
    property bool rightMarginClicked: false
    property bool topMarginClicked: false
    property bool bottomMarginClicked: false

    // single click and while being held
    onPressed: {
        if(mouse.button & Qt.LeftButton){
            if( borderRegion === "left" ){
                leftMarginClicked=true
                oldMouseX = mouseX
            } else if ( borderRegion === "right") {
                rightMarginClicked=true
                oldMouseX = mouseX
            } else if ( borderRegion === "top" ) {
                topMarginClicked=true
                oldMouseY = mouseY
            } else if ( borderRegion === "bottom") {
                bottomMarginClicked=true
                oldMouseY = mouseY
            }
        }
    }
    onReleased: {
        leftMarginClicked=false
        rightMarginClicked=false
        topMarginClicked=false
        bottomMarginClicked=false
    }

    onPositionChanged: {
        if (pressed) { // things to do while pressing the mouse key
            mouseDX = mouseX - oldMouseX
            oldMouseX = mouseX
            if(leftMarginClicked){
                blkRectId.width = blkRectId.width - mouseDX
                blkRectId.x = blkRectId.x + mouseDX
            }
        }
        model.blockXPosition = blkRectId.x
        model.blockYPosition = blkRectId.y
        xPosition = model.blockXPosition
        yPosition = model.blockYPosition
        flickableId.maxFlickX = Math.max(myBlockModel.maxBlockX() + width*2, flickableId.width)
        flickableId.maxFlickY = Math.max(myBlockModel.maxBlockY() + height*2, flickableId.height)
    }
}
