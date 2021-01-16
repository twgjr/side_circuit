import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0

Rectangle {
    id:eqRectId
    color: "lightblue"
    border.color: "black"
    border.width: 1
    radius: 1
    height: 25
    anchors.left: parent.left
    anchors.leftMargin: 5
    anchors.right: parent.right
    anchors.rightMargin: 5

    property string equationString: model.equationString

    MouseArea{
        id:equationMouseArea
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton

        onDoubleClicked: {
            if(mouse.button & Qt.LeftButton){
                equationDialog.open()
            }
        }
        // single click and release
        onClicked: {
            if(mouse.button & Qt.RightButton){
                equationContextMenu.popup()
            }
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

        Text{
            id: equationDisplayText
            anchors.centerIn: parent
            anchors.fill: parent
            text: equationString
        }

        Dialog{
            id:equationDialog
            TextField{
                id: equationText
                anchors.centerIn: parent
                anchors.fill: parent
                placeholderText: "Enter an equation"
            }
            onAccepted: {
                equationString = equationText.text
                model.equationString = equationString
                close()
            }
        }//Dialog
    }
}
