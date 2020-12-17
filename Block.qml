import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0

Rectangle{
    id:blkRectId
    property int idText: model.id
    property string descriptionText: model.description
    property int xPosition: model.blockXPosition
    property int yPosition: model.blockYPosition
    property string equationText: model.equationString

    color: "beige"
    border.color: "black"
    border.width: 2
    radius: 5
    height: 100; width:100
    x: xPosition
    y: yPosition
    Component.onCompleted: {
        //console.log(blockModel.thisBlock(model.index))
        //console.log(blockModel)
        //console.log(portModel)
        //console.log(model.index)
        //portModel.setBlockParent(blockModel.thisBlock(model.index))

    }
    Component.onDestruction: {}

    property int mouseBorder: 3*border.width
    BlockMouseArea{
        id: blockMouseAreaId
        anchors.centerIn: parent
        height: parent.height - mouseBorder
        width: parent.width - mouseBorder
    }
    BlockCornerMouseArea {
        id: bottomRightCornerId
        width: mouseBorder*2
        height: width
        anchors.horizontalCenter: parent.right
        anchors.verticalCenter: parent.bottom
    }

    PortModel{
        id: portModel
        blockDataSource: diagramDataSourceId.blockDataSource(model.index)
//        Component.onCompleted: {
////            console.log(blockModel.thisBlock(model.index))
////            console.log(blockModel)
////            console.log(portModel)
////            portModel.setBlockParent(blockModel.thisBlock(model.index))
//            console.log(model.thisBlock)
//            portModel.setBlockParent(model.thisBlock)
//        }
    }
    Repeater{
        id : portRepeater
        height: parent.height
        width: parent.width
        model : portModel
        delegate: Port{}
    }

    ColumnLayout{
        anchors.fill: parent
        RowLayout{
            Text {
                Layout.fillWidth: true
                text : "Block ID:" + idText
                horizontalAlignment: Text.AlignHCenter
            }
        }
    }
} //Rectangle

