import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import QtQuick.Shapes 1.15
import Qt.labs.qmlmodels 1.0
import com.company.models 1.0

Rectangle {
    id:eqRectId
    color: "lightblue"
    height:50
    width:50
    x: xPosition
    y: yPosition

    property int xPosition: model.xPos
    property int yPosition: model.yPos
    property string equationString: model.equationString

    Text{
        id: equationDisplayText
        anchors.centerIn: parent
        anchors.fill: parent
        text: equationString
    }
    MouseArea{
        id:equationMouseArea
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        drag.target: parent
        drag.threshold: 0
        property int posX
        property int posY

        onDoubleClicked: {
            if(mouse.button & Qt.LeftButton){
                //flickableId.leveltext = dataSource.distanceFromRoot()+1
                //dataSource.downLevel(model.index)
                equationDialog.open()
            }
        }
        // single click and release
        onClicked: {
            if(mouse.button & Qt.RightButton){
                equationContextMenu.popup()
            }
        }
        // single click and while being held
        onPressed: {}
        onReleased: {}
        onPositionChanged: {
            model.xPos = eqRectId.x
            model.yPos = eqRectId.y
            xPosition = model.xPos
            yPosition = model.yPos
            flickableId.maxFlickX = Math.max(dataSource.maxBlockX() + width*2, flickableId.width)
            flickableId.maxFlickY = Math.max(dataSource.maxBlockY() + height*2, flickableId.height)
        }

        Menu {
            id: equationContextMenu
            MenuItem {
                text: "Delete"
                onTriggered: {
                    dataSource.deleteEquation(model.index)
                }
            }
        } //Menu
        Dialog{
            id:equationDialog
            TextInput{
                id: equationText
                anchors.centerIn: parent
                anchors.fill: parent
                text: equationString
            }
            onAccepted: {
                equationString = equationText.text
                model.equationString = equationString
                close()
            }
        }//Dialog
    }
}
