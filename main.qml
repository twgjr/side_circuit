import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import QtQuick.Shapes 1.15
import com.company.models 1.0

Window {
    id: window
    visible: true
    width: 640
    height: 480
    title: qsTr("Diagram Solver")
    visibility: "Maximized"

    DataSource{ id: dataSource }

    ColumnLayout{
        id:columnLayoutId
        width: window.width
        height: window.height

        RowLayout{
            Text {
                id: textLevelId
                text : "Model Level: " + flickableId.leveltext
            }
        }//RowLayout

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
                property int leveltext: dataSource.distanceFromRoot()

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
                                flickableId.leveltext = Math.max(dataSource.distanceFromRoot()-1,0)
                                dataSource.upLevel()
                            }
                        }
                    }
                }//MouseArea

                BlockModel{
                    id: blockModel
                    dataSource: dataSource
                }
                Repeater{
                    id : repeaterID
                    height: parent.height
                    width: parent.width
                    model : blockModel
                    delegate: Block{}
                }
            } //Flickable
        } //RowLayout

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
                id : mButton6
                text : "Solve Model"
                Layout.fillWidth: true
                onClicked: {
                    dataSource.solveEquations()
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
            dataSource.saveBlockItems(fileSaveDialog.fileUrl)
            fileSaveDialog.close()
        }
        onRejected: {
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
            dataSource.loadBlockItems(fileLoadDialog.fileUrl)
            fileLoadDialog.close()
        }
        onRejected: {
            fileLoadDialog.close()
        }
    } //FileDialog
} //Window
