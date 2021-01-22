import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
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

    DiagramModel{
        id: diagramModel
        dataSource: dataSource
    }

    EquationModel{
        id: equationModel
        dataSource: dataSource
    }

    //Main window column layout
    ColumnLayout{
        anchors.fill: parent
        Layout.margins: 5
        // Top Menu
        RowLayout{
            spacing: 5

            Button{
                id: projectMenuButton
                Layout.margins: 5
                text: "\u2261"
                Layout.preferredWidth: 50
                Layout.preferredHeight: 50
                font.pointSize: 15
                hoverEnabled: true
                ToolTip.delay: 1000
                ToolTip.timeout: 3000
                ToolTip.visible: hovered
                ToolTip.text: "Menu"
                onClicked: {
                    projectMenu.open()
                }

                Menu{
                    id:projectMenu
                    y:projectMenuButton.height+5
                    MenuItem{
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
                id : solveButton
                text : "🖩"
                bottomPadding: 15
                // unicode pocket calculator
                Layout.preferredWidth: 50
                Layout.preferredHeight: 50
                font.pointSize: 30
                hoverEnabled: true
                ToolTip.delay: 1000
                ToolTip.timeout: 3000
                ToolTip.visible: hovered
                ToolTip.text: "Solve"
                onClicked: {
                    dataSource.solveEquations()
                }
            }
            Button {
                id : rotateLeftButton
                Layout.margins: 5
                text : "⭯"
                Layout.preferredWidth: 50
                Layout.preferredHeight: 50
                font.pointSize: 15
                hoverEnabled: true
                ToolTip.delay: 1000
                ToolTip.timeout: 3000
                ToolTip.visible: hovered
                ToolTip.text: "Rotate Counter Clockwise"
                //onClicked signal handler in element or block connection
            }
            Button {
                id : rotateRightButton
                Layout.margins: 5
                text : "⭮"
                Layout.preferredWidth: 50
                Layout.preferredHeight: 50
                font.pointSize: 15
                hoverEnabled: true
                ToolTip.delay: 1000
                ToolTip.timeout: 3000
                ToolTip.visible: hovered
                ToolTip.text: "Rotate Clockwise"
                //onClicked signal handler in element or block connection
            }
        } //RowLayout

        RowLayout{// row containing diagram area and equation list
            spacing:5



            Rectangle{
                Layout.fillHeight: true
                Layout.fillWidth: true
                Layout.margins: 5

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
                        acceptedButtons: Qt.RightButton | Qt.LeftButton
                        anchors.fill: parent
                        hoverEnabled: true

                        onClicked: {
                            if(mouse.button & Qt.RightButton){
                                diagramAreaContextMenu.popup()
                            }
                            if(mouse.button & Qt.LeftButton){
                                flickableId.dItemIsSelected = []
                            }
                        }
                        Menu {
                            id: diagramAreaContextMenu
                            Menu {
                                title: "New Diagram Item"
                                MenuItem {
                                    text: "Block"
                                    onTriggered: dataSource.appendDiagramItem(
                                                     DItemTypes.BlockItem,
                                                     diagramMouseArea.mouseX,
                                                     diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "Resistor"
                                    onTriggered: dataSource.appendDiagramItem(
                                                     DItemTypes.Resistor,
                                                     diagramMouseArea.mouseX,
                                                     diagramMouseArea.mouseY)
                                }
                            }

                            MenuItem {
                                text: "Up Level"
                                onTriggered: {
                                    dataSource.upLevel()

                                }
                            }
                        }
                    }//MouseArea

                    property var dItemIsSelected: []

                    Repeater{
                        id : diagramRepeater
                        model : diagramModel
                        delegate: DiagramItem{}
                    }
                }//Flickable
            }//Rectangle


            ResultModel{
                id: resultModel
                dataSource: dataSource
            }


            ColumnLayout{
                Layout.bottomMargin: 0
                Layout.topMargin: 0
                spacing: 0
                Layout.margins: 5

                Rectangle{
                    id: equationListBorder
                    width:200
                    Layout.margins: 5
                    border.color: "black"
                    border.width: 1
                    Layout.fillHeight: true

                    ColumnLayout {
                        id: columnLayout
                        anchors.fill: parent

                        Button{
                            Layout.margins: 5
                            Layout.alignment: Qt.AlignHCenter
                            text: "Add Equation"
                            Layout.fillWidth: true
                            onClicked: {
                                dataSource.addEquation()
                            }
                        }

                        Label{ text: "Equations" ; horizontalAlignment: Text.AlignHCenter; Layout.fillWidth: true; }

                        ListView{
                            id : equationListView
                            Layout.fillHeight: true
                            Layout.fillWidth: true
                            anchors.topMargin: 5
                            anchors.bottomMargin: 5
                            spacing: 5
                            model : equationModel
                            delegate: Equation{}
                        }
                    }
                }

                Rectangle{
                    id: resultListBorder
                    border.color: "black"
                    border.width: 1
                    Layout.preferredHeight: 200
                    Layout.margins: 5
                    Layout.preferredWidth: 200

                    ColumnLayout {
                        id: columnLayout1
                        anchors.fill: parent
                        Label{
                            id: resultsTitleLabel
                            text: "Results"
                            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
                        }
                        RowLayout{
                            Layout.fillWidth: true
                            Label{
                                id: resultsVariableColLabel
                                text: "Variable"
                                horizontalAlignment: Text.AlignHCenter
                                Layout.fillWidth: true
                            }
                            Label{
                                id: resultsValueColLabel
                                text: "Value"
                                horizontalAlignment: Text.AlignHCenter
                                Layout.fillWidth: true
                            }
                        }

                        ListView{
                            id : resultListView
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            spacing: 5
                            model : resultModel
                            delegate: Result{}
                        }
                    }
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

/*##^##
Designer {
    D{i:0;formeditorZoom:0.75}D{i:42}D{i:48}
}
##^##*/
