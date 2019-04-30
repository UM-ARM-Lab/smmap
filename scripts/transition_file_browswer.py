#! /usr/bin/env python

import sys
import rospy
import signal
from copy import deepcopy
from deformable_manipulation_msgs.srv import TransitionTestingVisualization
from IPython import embed


from PyQt5.QtWidgets import *
from PyQt5.QtGui import QStandardItem, QStandardItemModel


class Widget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.data_folder = rospy.get_param("transition_learning_data_generation_node/data_folder", "/tmp")
        self.last_folder_used = deepcopy(self.data_folder)

        self.createSourceFileBox()
        self.createTestFileBox()
        self.createVisIdFeedbackBox()

        self.layout = QGridLayout()
        self.layout.addWidget(self.source_file_box)
        self.layout.addWidget(self.test_file_box)
        self.layout.addWidget(self.vis_id_box)

        self.setLayout(self.layout)
        self.resize(1000, 700)

    def createSourceFileBox(self):
        self.source_file_box = QGroupBox("Source Transition")

        text_field = QLineEdit()
        set_button = QPushButton("Set")
        browse_button = QPushButton("Browse")

        layout = QGridLayout()
        layout.addWidget(text_field, 0, 0)
        layout.addWidget(set_button, 0, 1)
        layout.addWidget(browse_button, 0, 2)
        self.source_file_box.setLayout(layout)

        browse_button.clicked.connect(lambda: self.fileBrowse(text_field, self.last_folder_used))
        set_button.clicked.connect(lambda: self.setSourceTransition(text_field))

    def createTestFileBox(self):
        self.test_file_box = QGroupBox("Test Transition")

        text_field = QLineEdit()
        add_button = QPushButton("Add")
        browse_button = QPushButton("Browse")

        layout = QGridLayout()
        layout.addWidget(text_field, 0, 0)
        layout.addWidget(add_button, 0, 1)
        layout.addWidget(browse_button, 0, 2)
        self.test_file_box.setLayout(layout)

        browse_button.clicked.connect(lambda: self.fileBrowse(text_field, self.last_folder_used))
        add_button.clicked.connect(lambda: self.addVisualization(text_field))

    def createVisIdFeedbackBox(self):
        self.vis_id_box = QGroupBox("Visualized Adapatations")

        self.ID_IDX = 0
        self.DELETE_IDX = 1
        self.FILENAME_IDX = 2

        self.vis_id_model = QStandardItemModel(0, 3)
        self.vis_id_model.setHorizontalHeaderLabels(["Vis ID", "Delete", "Filename"])

        self.vis_id_view = QTreeView(self)
        self.vis_id_view.setModel(self.vis_id_model)
        self.vis_id_view.setAlternatingRowColors(True)
        self.vis_id_view.setSortingEnabled(True)
        self.vis_id_view.setColumnWidth(self.ID_IDX, 50)
        self.vis_id_view.setColumnWidth(self.DELETE_IDX, 75)

        layout = QGridLayout()
        layout.addWidget(self.vis_id_view)
        self.vis_id_box.setLayout(layout)

    def fileBrowse(self, text_field, directory=None):
        file, filter = QFileDialog.getOpenFileName(parent=self, directory=directory, filter="*.compressed")
        text_field.setText(file[len(self.data_folder) + 1:])

    def setSourceTransition(self, text_field):
        rospy.wait_for_service("transition_vis/set_source")
        try:
            set_source = rospy.ServiceProxy("transition_vis/set_source", TransitionTestingVisualization)
            set_source(text_field.text())
        except rospy.ServiceException, e:
            print "set_source service call failed: ", e

    def addVisualization(self, text_field):
        rospy.wait_for_service("transition_vis/add_visualization")
        try:
            add_visualization = rospy.ServiceProxy("transition_vis/add_visualization", TransitionTestingVisualization)
            filename = text_field.text()
            id = add_visualization(filename).response

            id_item = QStandardItem(id)
            filename_item = QStandardItem(filename)
            self.vis_id_model.appendRow([id_item, QStandardItem(), filename_item])

            delete_button = QPushButton()
            delete_button.setText("Delete")
            delete_button.clicked.connect(lambda: self.removeVisualization(id))
            delete_button_index = self.vis_id_model.index(id_item.row(), self.DELETE_IDX)
            self.vis_id_view.setIndexWidget(delete_button_index, delete_button)

        except rospy.ServiceException, e:
            print "add_visualization service call failed: ", e

    def removeVisualization(self, id):
        rospy.wait_for_service("transition_vis/remove_visualization")
        try:
            remove_visualization = rospy.ServiceProxy("transition_vis/remove_visualization", TransitionTestingVisualization)
            remove_visualization(id)

        except rospy.ServiceException, e:
            print "remove_visualization service call failed: ", e


if __name__ == "__main__":
    rospy.init_node("transition_file_browser", anonymous=True)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    widget = Widget()
    widget.setWindowTitle("Data Visualization")
    widget.show()

    sys.exit(app.exec_())