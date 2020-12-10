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

        Text {
            id: distanceFromRootID
            Layout.fillWidth: true
            text : "Distance From Root:" + myBlockModel.distanceFromRoot()
            horizontalAlignment: Text.AlignLeft
        }

        // wrap Repeater in flickable to enable both h and v panning
        RowLayout{
            id: flickableRowLayoutId
            Layout.fillHeight: true
            Layout.fillWidth: true

            Flickable{
                id:flickableId
                property int maxFlickX: Math.max(myBlockModel.maxBlockX()+width*2,parent.width)
                property int maxFlickY: Math.max(myBlockModel.maxBlockY()+height*2,parent.height)

                flickableDirection: Flickable.HorizontalAndVerticalFlick
                height: parent.height
                width: parent.width
                contentWidth: maxFlickX
                contentHeight: maxFlickY
                clip: true
                ScrollBar.horizontal: ScrollBar { id: hbar ; active: true; visible: true ; policy: ScrollBar.AsNeeded }
                ScrollBar.vertical: ScrollBar { id: vbar; active: true; visible: true ; policy: ScrollBar.AsNeeded }

                MouseArea{
                    acceptedButtons: Qt.RightButton
                    anchors.fill: parent

                    onClicked: {
                        if(mouse.button & Qt.RightButton){
                            myBlockModel.upLevel()
                        }
                    }
                }

                Repeater{
                    height: parent.height
                    width: parent.width
                    id : repeaterID
                    model : myBlockModel
                    delegate: Rectangle{
                        id: rectangleDelegateId
                        property int idText: 0
                        property string categoryText: ""
                        property int xPosition: 0
                        property int yPosition: 0
                        property string equationText: ""
                        property int numChildrenText: 0
                        property bool selected: false



                        color: "beige"
                        border.color: "yellowgreen"
                        radius: 5
                        height: 250; width:250
                        x: xPosition
                        y: yPosition


                        Component.onCompleted: {
                            xPosition = model.blockXPosition
                            yPosition = model.blockYPosition
                            idText = model.id
                            categoryText = model.category
                            equationText = model.equationString
                            numChildrenText = myBlockModel.numChildren(model.index)
                            distanceFromRootID.text = "Distance From Root:" + myBlockModel.distanceFromRoot()
                        }

                        onXChanged: {
                            model.blockXPosition = rectangleDelegateId.x
                            xPosition = model.blockXPosition
                            flickableId.maxFlickX = myBlockModel.maxBlockX()+width*2
                        }
                        onYChanged: {
                            model.blockYPosition = rectangleDelegateId.y
                            yPosition = model.blockYPosition
                            flickableId.maxFlickY = myBlockModel.maxBlockY() + height*2
                        }
                        Component.onDestruction: {
                            distanceFromRootID.text = "Distance From Root:" + myBlockModel.distanceFromRoot()
                        }

                        MouseArea{
                            acceptedButtons: Qt.LeftButton | Qt.RightButton
                            anchors.fill: parent
                            drag.target: parent

                            onDoubleClicked: {
                                print("Double Clicked: "+model.index)
                                myBlockModel.downLevel(model.index)

                            }
                        }

                        ColumnLayout{
                            anchors.fill: parent
                            RowLayout{

                                Text {
                                    Layout.fillWidth: true
                                    text : "Block ID:" + idText
                                    horizontalAlignment: Text.AlignHCenter
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text : "# Children:" + numChildrenText
                                    horizontalAlignment: Text.AlignHCenter
                                }
                            }


                            TextField {
                                Layout.fillWidth: true
                                placeholderText: "Enter an category"
                                text : categoryText
                                horizontalAlignment: Text.AlignHCenter
                                onEditingFinished: {
                                    model.category = text
                                    categoryText = model.category
                                }
                            }
                            Text {
                                Layout.fillWidth: true
                                text : xPosition + " x " + yPosition
                                horizontalAlignment: Text.AlignHCenter
                            }
                            TextField {
                                Layout.fillWidth: true
                                placeholderText: "Enter an equation"
                                text: equationText
                                horizontalAlignment: Text.AlignHCenter
                                onEditingFinished: {
                                    model.equationString = text
                                    equationText = model.equationString
                                }
                            }
                            Button {
                                text : "Print C++ Model"
                                Layout.fillWidth: true
                                onClicked: {
                                    myBlockModel.printBlock(index)
                                }
                            }
                            Button {
                                text : "Print Proxy Model"
                                Layout.fillWidth: true
                                onClicked: {
                                    print("ID: " + model.id)
                                    print("Category: " + model.category)
                                    print("Position: " + model.blockXPosition + " x " + model.blockYPosition)
                                    print("Equation: " + model.equationString)
                                }
                            }
                            Button {
                                text : "Print QML values"
                                Layout.fillWidth: true
                                onClicked: {
                                    print("ID: " + parent.parent.idText)
                                    print("Category: " + parent.parent.categoryText)
                                    print("Position: " + parent.parent.xPosition + " x " + parent.parent.yPosition)
                                    print("Equation: " + parent.parent.equationText)
                                }
                            }
                        }
                    }
                }
            }
        }
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
                id : mButton2
                text : "RemoveLast"
                Layout.fillWidth: true
                onClicked: myBlockModel.removeLastBlockItem()
            }
            Button {
                id : mButton3
                text : "Add New Block"
                Layout.fillWidth: true
                onClicked: myBlockModel.appendBlock() // adds to the active "root"
            }
            Button {
                id : mButton4
                text : "Save"
                Layout.fillWidth: true
                onClicked: fileSaveDialog.open()
            }
            Button {
                id : mButton5
                text : "Clear"
                Layout.fillWidth: true
                onClicked: myBlockModel.clearBlockItems()
            }
            Button {
                id : mButton6
                text : "Solve Model"
                Layout.fillWidth: true
                onClicked: {
                    myBlockModel.solveEquations()
                }
            }
        }
    }


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
    }
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
    }
}
