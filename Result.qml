import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0

Rectangle {
    id:resultRectId
    color: "lightblue"
    border.color: "black"
    border.width: 1
    radius: 1
    height: 25
    anchors.left: parent.left
    anchors.right: parent.right

    property string varName: model.varName
    property double varVal: model.varVal

    MouseArea{
        id:resultMouseArea
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton

        // single click and release
        onClicked: {
            if(mouse.button & Qt.RightButton){
                resultContextMenu.popup()
            }
        }

        Menu {
            id: resultContextMenu
            MenuItem {
                text: "Dummy"
                onTriggered: {
                    //do somthing with the result
                }
            }
        } //Menu

    }
    RowLayout{
        anchors.fill: parent
        Text{
            id: varDisplayText
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            Layout.preferredWidth: parent.width/2
            text: varName
        }
        Text{
            id: resultDisplayText
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            Layout.preferredWidth: parent.width/2
            text: varVal
        }
    }
}