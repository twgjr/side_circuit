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

            DiagramModel{
                id: diagramModel
                dataSource: dataSource
            }
            ElementModel{
                id: elementModel
                dataSource: dataSource
            }

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

                        onClicked: {
                            if(mouse.button & Qt.RightButton){
                                diagramAreaContextMenu.popup()
                            }
                            if(mouse.button & Qt.LeftButton){
                                flickableId.blockIsSelected = []
                                flickableId.elementIsSelected = []
                            }
                        }
                        Menu {
                            id: diagramAreaContextMenu
                            Menu {
                                title: "New Element"
                                MenuItem {
                                    text: "Generic"
                                    onTriggered: dataSource.addElement(0,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "Resistor"
                                    onTriggered: dataSource.addElement(1,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "Capacitor"
                                    onTriggered: dataSource.addElement(2,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "Inductor"
                                    onTriggered: dataSource.addElement(3,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "Solid State Switch"
                                    onTriggered: dataSource.addElement(4,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "Diode"
                                    onTriggered: dataSource.addElement(5,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "IC"
                                    onTriggered: dataSource.addElement(6,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "LED"
                                    onTriggered: dataSource.addElement(7,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "Battery"
                                    onTriggered: dataSource.addElement(8,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                                MenuItem {
                                    text: "Connector"
                                    onTriggered: dataSource.addElement(9,diagramMouseArea.mouseX,diagramMouseArea.mouseY)
                                }
                            }
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

                    property var blockIsSelected: []
                    property var elementIsSelected: []

                    Repeater{
                        id : blockRepeater
                        model : diagramModel
                        delegate: Block{}
                    }
                    Repeater{
                        id : elementRepeater
                        model : elementModel
                        delegate: Element{}
                    }
                }//Flickable
            }//Rectangle


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

                Rectangle{
                    id: equationListBorder
                    width:200
                    Layout.margins: 5
                    Layout.fillHeight: true
                    border.color: "black"
                    border.width: 1

                    ListView{
                        id : equationListView
                        anchors.fill: parent
                        anchors.topMargin: 5
                        anchors.bottomMargin: 5
                        spacing: 5
                        model : equationModel
                        delegate: Equation{}
                    }
                }

                Rectangle{
                    id: resultListBorder
                    width:200
                    Layout.margins: 5

                    Layout.fillHeight: true
                    border.color: "black"
                    border.width: 1

                    ColumnLayout{
                        Label{
                            text: "Results"
                            Layout.fillWidth: true
                        }
                        RowLayout{
                            Label{
                                text: "Variable"
                                Layout.fillWidth: true
                            }
                            Label{
                                text: "Value"
                                Layout.fillWidth: true
                            }
                        }
                    }
                    ListView{
                        id : resultListView
                        anchors.fill: parent
                        anchors.topMargin: 5
                        anchors.bottomMargin: 5
                        spacing: 5
                        model : resultModel
                        delegate: Result{}
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
