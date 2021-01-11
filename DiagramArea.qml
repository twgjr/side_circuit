import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import QtQuick.Shapes 1.15
import com.company.models 1.0

Rectangle{
    id: diagramArea
    border.color: "black"
    border.width: 1


    Flickable{
        id:flickableId
        property int maxFlickX: width
        property int maxFlickY: height

        flickableDirection: Flickable.HorizontalAndVerticalFlick
        height: parent.height
        width: parent.width

        contentWidth: maxFlickX
        contentHeight: maxFlickY
        clip: true
        ScrollBar.horizontal: ScrollBar {
            id: hbar
            active: true; visible: true
            policy: ScrollBar.AlwaysOn
        }
        ScrollBar.vertical: ScrollBar {
            id: vbar
            active: true; visible: true
            policy: ScrollBar.AlwaysOn
        }

        MouseArea{
            id: diagramMouseArea
            acceptedButtons: Qt.RightButton
            anchors.fill: parent

            onClicked: {
                if(mouse.button & Qt.RightButton){
                    diagramAreaContextMenu.popup()
                }
            }
            Menu {
                id: diagramAreaContextMenu
                MenuItem {
                    text: "New Block"
                    onTriggered: dataSource.appendBlock(diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                }
                MenuItem {
                    text: "Up Level"
                    onTriggered: {
                        dataSource.upLevel()
                    }
                }
            }
        }//MouseArea

        Repeater{
            id : blockRepeater
            model : diagramModel
            delegate: Block{}
        }
    }//Flickable
}//Rectangle
