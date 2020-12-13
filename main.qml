import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0

Window {
    id: window
    visible: true
    width: 640
    height: 480
    title: qsTr("Diagram Solver")
    visibility: "Maximized"

    BlockModel{id:myBlockModel}

    ColumnLayout{
        id:columnLayoutId
        width: window.width
        height: window.height

        // wrap Repeater in flickable to enable both h and v panning
        RowLayout{
            id: flickableRowLayoutId
            Layout.fillHeight: true
            Layout.fillWidth: true

            // the diagram area
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
                ScrollBar.horizontal: ScrollBar { id: hbar ; active: true; visible: true ; policy: ScrollBar.AsNeeded }
                ScrollBar.vertical: ScrollBar { id: vbar; active: true; visible: true ; policy: ScrollBar.AsNeeded }

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
                            onTriggered: myBlockModel.appendBlock(diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                        }
                        MenuItem {
                            text: "Up Level"
                            onTriggered: myBlockModel.upLevel()
                        }
                        MenuItem {
                            text: "Set Background Color"
                            onTriggered: diagramColorDialog.open()
                        }
                    }
                    ColorDialog {
                        id: diagramColorDialog
                        currentColor: rectangleDelegateId.color
                        title: "Please choose a block color"
                        onAccepted: {
                            console.log("You chose: " + color)
                            rectangleDelegateId.color = color
                            close()
                        }
                        onRejected: {
                            console.log("Canceled")
                            close()
                        }
                    }
                }

                Repeater{
                    id : repeaterID
                    height: parent.height
                    width: parent.width
                    model : myBlockModel
                    delegate: QBlock{}
                }
            } //Flickable
        } //RowLayout
        RowLayout{
            Frame{
                Layout.fillWidth: parent
                ColumnLayout {
                    Label{text: "Solver Results"}
                    ScrollView{
                        TextArea{
                            placeholderText: "Solver Result Displays Here"
                        }
                    }
                }
            }
        }

        RowLayout{
            id: buttonRow
            Layout.fillHeight: true
            Layout.fillWidth: true
            Button {
                id : mButton1
                text : "Open"
                Layout.fillWidth: true
                onClicked: fileLoadDialog.open()
            }
            Button {
                id : mButton4
                text : "Save"
                Layout.fillWidth: true
                onClicked: fileSaveDialog.open()
            }
            Button {
                id : mButton3
                text : "New Block"
                Layout.fillWidth: true
                onClicked: myBlockModel.appendBlock(window.width/2,window.height/2) // adds to the active "root"
            }
            Button {
                id : mButton6
                text : "Solve Model"
                Layout.fillWidth: true
                onClicked: {
                    myBlockModel.solveEquations()
                }
            }
        } //RowLayout
    } //ColumnLayout


    FileDialog {
        id: fileSaveDialog
        title: "Please save the acive model to JSON file"
        folder: shortcuts.home
        selectMultiple: false
        selectExisting: false
        nameFilters: [ "JSON file (*.json)" ]
        onAccepted: {
            myBlockModel.saveBlockItems(fileSaveDialog.fileUrl)
            fileSaveDialog.close()
        }
        onRejected: {
            console.log("Canceled")
            fileSaveDialog.close()
        }
    }//FileDialog
    FileDialog {
        id: fileLoadDialog
        title: "Please load the JSON model file"
        folder: shortcuts.home
        selectMultiple: false
        selectExisting: true
        nameFilters: [ "JSON file (*.json)" ]
        onAccepted: {
            myBlockModel.loadBlockItems(fileLoadDialog.fileUrl)
            fileLoadDialog.close()
        }
        onRejected: {
            console.log("Canceled")
            fileLoadDialog.close()
        }
    } //FileDialog
} //Window
