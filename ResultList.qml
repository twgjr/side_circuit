import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import QtQuick.Shapes 1.15
import com.company.models 1.0

Rectangle{
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
