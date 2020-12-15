import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0
import "portCreation.js" as PortScript

Rectangle{
    id: blkRectId
    property int idText: model.id
    property string descriptionText: model.description
    property int xPosition: model.blockXPosition
    property int yPosition: model.blockYPosition
    property string equationText: model.equationString
    property bool selected: false

    color: "beige"
    border.color: "black"
    border.width: 2
    radius: 5
    height: 100; width:100
    x: xPosition
    y: yPosition

    transform: Scale{id:blockScaleId}

    Component.onCompleted: {
        var portCount = myBlockModel.portCount(model.index)
        for( var i = 0 ; i < portCount ; i++){
            var side = myBlockModel.portSide(model.index,i)
            var position = myBlockModel.portPosition(model.index,i)
            PortScript.createPortObjects(side,position);
        }
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
