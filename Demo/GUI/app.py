import sys
import logging
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLineEdit, QTextEdit, QFileDialog
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from threading import Event  # Import Event for cancellation
from Model_test import process_log_in_intervals

# Worker thread class to handle background processing
class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    resource_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(float, float)  # Signal to indicate that the processing is finished with memory and CPU usage

    def __init__(self, log_file, model_path, interval_minutes, label_mapping_choice, stop_event):
        super().__init__()
        self.log_file = log_file
        self.model_path = model_path
        self.interval_minutes = interval_minutes
        self.label_mapping_choice = label_mapping_choice  # Add this to store the label mapping choice
        self.stop_event = stop_event  # Add stop_event to the thread

    def run(self):
        # Call process_log_in_intervals and capture both predictions and resource usage
        max_memory, max_cpu = process_log_in_intervals(
            self.log_file, 
            self.model_path, 
            self.interval_minutes,
            self.log_signal, 
            self.resource_signal, 
            stop_event=self.stop_event, 
            label_mapping_choice=self.label_mapping_choice  # Pass the label mapping choice
        )
        self.finished_signal.emit(max_memory, max_cpu)  # Emit the finished signal with memory and CPU usage

class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit, filter_type=None):
        super().__init__()
        self.text_edit = text_edit
        self.filter_type = filter_type  # "resource" or "result"

    def emit(self, record):
        msg = self.format(record)
        
        # Filter based on the message content for proper logging
        if self.filter_type == "resource" and "Resource Usage" in msg:
            self.text_edit.append(msg)
        elif self.filter_type == "result" and "Batch" in msg:
            self.text_edit.append(msg)

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None  # Initialize worker to None
        self.stop_event = Event()  # Event to signal cancellation
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Demo Project Testing")
        self.setGeometry(200, 200, 800, 600)

        # Set dark theme colors
        self.setStyleSheet(""" 
            QWidget {
                background-color: #2E2E2E;
                color: #FFFFFF;
            }
            QLabel {
                font-size: 18px; 
                font-weight: bold;  /* Set font to bold */
            }
            QPushButton {
                background-color: #3D3D3D;
                border: none;
                color: #FFFFFF;
                font-family: 'Arial';
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton#testButton {
                background-color: #28A745;  /* Green color for Test button */
            }
            QPushButton#cancelButton {
                background-color: #ffae00;  /* Yellow color for Cancel button */
            }
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFFFFF;
                font-size: 14px;
            }
            QComboBox {
                font-size: 16px;  /* Set font size for dropdowns */
                padding-left: 10px;
            }
            QLineEdit {
                font-size: 16px;
            }
        """)
        
        # Layout
        layout = QVBoxLayout()

        # Title label
        title_label = QLabel("Project II Demo")
        title_font = QFont('Arial', 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)  # Align text to the center
        layout.addWidget(title_label)

        # Log file selection
        log_layout = QHBoxLayout()
        log_label = QLabel("Log File:")
        self.log_input = QLineEdit(self)
        self.log_input.setFixedSize(400, 40)

        log_browse_button = QPushButton("Browse", self)
        log_browse_button.clicked.connect(self.browse_log_file)
        log_browse_button.setFixedSize( 200, 40)

        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_input)
        log_layout.addWidget(log_browse_button)

        # Model selection dropdown
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_dropdown = QComboBox(self)
        self.model_dropdown.setFixedSize(200, 40)
        self.model_dropdown.addItems(['Decision tree', 'SVM', 'Random Forest', 'Adaboost', 'Demo-DCT', 'Demo-SVM', 'Demo-RF', 'Demo-AB'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)

        # Interval selection dropdown
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Interval:")
        self.interval_dropdown = QComboBox(self)
        self.interval_dropdown.setFixedSize(200, 40)
        self.interval_dropdown.addItems(['1 minute', '5 minutes', '15 minutes', '30 minutes', '1 hour', '2 hours', '4 hours', '8 hours', '16 hours', '24 hours'])
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_dropdown)

        # Button to start testing
        self.test_button = QPushButton("Test", self)
        self.test_button.setFixedSize(200, 40)
        self.test_button.setObjectName("testButton")  # Set object name for styling
        self.test_button.clicked.connect(self.start_test)

        # Button to cancel the testing
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setFixedSize(200, 40)
        self.cancel_button.setObjectName("cancelButton")  # Set object name for styling
        self.cancel_button.clicked.connect(self.cancel_test)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.cancel_button)

        # Set the alignment of the buttons to center
        button_layout.setAlignment(Qt.AlignCenter)

        # Output Field for Resource Usage (small box)
        self.resource_usage_area = QTextEdit(self)
        self.resource_usage_area.setReadOnly(True)
        self.resource_usage_area.setFixedHeight(100)  # Set smaller height for resource usage

        # Output Field for Test Results (large box)
        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        self.result_area.setMinimumHeight(300)

        layout.addLayout(log_layout)
        layout.addLayout(model_layout)
        layout.addLayout(interval_layout)
        # layout.addWidget(self.test_button)
        # layout.addWidget(self.cancel_button)  # Add the cancel button
        layout.addLayout(button_layout)
        layout.addWidget(QLabel("Resource Usage Output:"))
        layout.addWidget(self.resource_usage_area)
        layout.addWidget(QLabel("Test Results Output:"))
        layout.addWidget(self.result_area)

        self.setLayout(layout)

        # Setup logging to display on GUI
        self.setup_logging()

    def setup_logging(self):
        """Setup custom logging handlers for displaying output in QTextEdit fields."""
        # Handler for resource usage
        resource_usage_handler = QTextEditLogger(self.resource_usage_area, filter_type="resource")
        resource_usage_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        resource_usage_handler.setLevel(logging.INFO)

        # Handler for test results
        test_result_handler = QTextEditLogger(self.result_area, filter_type="result")
        test_result_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        test_result_handler.setLevel(logging.INFO)

        # Add handlers to the logger
        logging.getLogger().addHandler(resource_usage_handler)
        logging.getLogger().addHandler(test_result_handler)
        logging.getLogger().setLevel(logging.INFO)

    def browse_log_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'Log Files (*.log *.txt *.labeled)')
        if file_name:
            self.log_input.setText(file_name)

    def start_test(self):
    # If a test is currently running, cancel it before starting a new one
        if self.worker is not None:
            self.stop_event.set()  # Stop the current test
            self.worker.wait()  # Wait for the thread to finish

        # Clear output areas
        self.resource_usage_area.clear()
        self.result_area.clear()

        log_file = self.log_input.text()
        model_choice = self.model_dropdown.currentText()
        interval_choice = self.interval_dropdown.currentText()

        model_map = {
            # use 1st Label Mapping
            'Decision tree': 'Model - DCT-mali.joblib',
            'SVM': 'Model - SVM-mali.joblib',
            'Random Forest': 'Model - RF-mali.joblib',
            'Adaboost': 'Model - AdaBoost-mali.joblib',

            # use 2nd Label Mapping
            'Demo-DCT': 'Model - DCT.joblib',
            'Demo-SVM': 'Model - SVM.joblib',
            'Demo-RF': 'Model - RF.joblib',
            'Demo-AB': 'Model - AdaBoost-Tuned.joblib',
        }
    
        # Determine whether to use 1st or 2nd label mapping
        if 'Demo' in model_choice:
            label_mapping_choice = 2  # Use the 2nd label mapping
        else:
            label_mapping_choice = 1  # Use the 1st label mapping

        model_path = f"C:/Users/Natty PC/Documents/Party/Project II/Models/{model_map[model_choice]}"

        # Map the interval choices to minutes
        interval_map = {
            '1 minute': 1,
            '5 minutes': 5,
            '15 minutes': 15,
            '30 minutes': 30,
            '1 hour': 60,
            '2 hours': 120,
            '4 hours': 240,
            '8 hours': 480,
            '16 hours': 960,
            '24 hours': 1440
        }

        interval_minutes = interval_map.get(interval_choice, 5)  # Default to 5 minutes if not found

        self.result_area.append(f"Starting test with model: {model_choice}, interval: {interval_choice}")
        logging.info(f"Starting test for log file: {log_file} with model: {model_choice} and interval: {interval_minutes} minutes")

        # Reset the stop_event for the new test
        self.stop_event.clear()

        # Start worker thread for background processing
        self.worker = WorkerThread(log_file, model_path, interval_minutes, label_mapping_choice, self.stop_event)
        self.worker.log_signal.connect(self.update_test_output)
        self.worker.resource_signal.connect(self.update_resource_output)
        self.worker.finished_signal.connect(self.finish_testing)  # Connect the finished signal
        self.worker.start()

    def cancel_test(self):
        """Cancel the ongoing test."""
        if self.worker is not None:
            self.stop_event.set()  # Set the stop event to signal cancellation
            self.result_area.append("Testing cancelled.")
            logging.info("Testing cancelled by user.")

    def finish_testing(self, max_memory, max_cpu):
        """Update the result area to indicate that testing is finished."""
        self.result_area.append("Finish Testing.")
        logging.info("Testing finished.")
        
        # Append memory and CPU usage to resource usage area
        self.resource_usage_area.append(f"Max Memory Used During Execution: {max_memory:.2f} MB")
        self.resource_usage_area.append(f"Max CPU Used During Execution: {max_cpu:.2f}%")

    def update_test_output(self, message):
        """Update the test result QTextEdit with new message"""
        self.result_area.append(message)

    def update_resource_output(self, message):
        """Update the resource usage QTextEdit with new message"""
        self.resource_usage_area.append(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
