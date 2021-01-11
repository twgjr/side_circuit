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
    width: 1280
    height: 720
    title: qsTr("Diagram Solver")
    visibility: "Maximized"

    DataSource{ id: dataSource }

    //Main window column layout
    ColumnLayout{
        anchors.fill: parent
        Layout.margins: 5
        // Top Menu
        RowLayout{
            spacing: 5

            MenuBar{
                Layout.margins: 5
                Menu{
                    title: "File"
                    Action{
                        text : "Open"
                        onTriggered: fileLoadDialog.open()
                    }
                    MenuItem{
                        text : "Save"
                        onTriggered: fileSaveDialog.open()
                    }
                }
            }
            ToolSeparator{}
            Button {
                Layout.margins: 5
                id : solveButton
                text : "Solve Model"
                onClicked: {
                    dataSource.solveEquations()
                }
            }
        } //RowLayout

        RowLayout{// row containing diagram area and equation list
            spacing:5

            DiagramModel{
                id: diagramModel
                dataSource: dataSource
            }
            DiagramArea{
                Layout.fillHeight: true
                Layout.fillWidth: true
                Layout.margins: 5
            }

            EquationModel{
                id: equationModel
                dataSource: dataSource
            }
            ResultModel{
                id: resultModel
                dataSource: dataSource
            }

            ColumnLayout{
                Button{
                    Layout.margins: 5
                    Layout.alignment: Qt.AlignHCenter
                    Layout.preferredWidth: equationListBorder.width
                    text: "Add Equation"
                    onClicked: {
                        dataSource.addEquation()
                    }
                }
                Label{text: "Equations"}
                EquationList{
                    id: equationListBorder
                    width:200
                    Layout.margins: 5
                }

                ResultList{
                    id: resultListBorder
                    width:200
                    Layout.margins: 5
                }
            }
        }//RowLayout
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
