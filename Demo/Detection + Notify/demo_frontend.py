# demo_frontend.py

import logging
import tkinter as tk
from tkinter import filedialog
from demo_backend import process_log_in_intervals  # Assuming the process_log_in_intervals function is defined in demo_backend.py
from threading import Event

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fixed model path
MODEL_PATH = "C:/Users/USER/Documents/Party/01 - PSU/Project II/Models/Model - DCT.joblib"

def browse_log_file():
    """Allow the user to browse and select a log file."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window to avoid displaying the tkinter main window
    
    # Default directory for log files
    default_log_dir = "C:/Users/USER/Documents/Party/01 - PSU/Project II/Data/"
    
    # Ask user to select the log file, starting in the default directory
    file_path = filedialog.askopenfilename(
        title="Select Log File", 
        filetypes=(("All files", "*.*"), ("Text files", "*.txt")),
        initialdir=default_log_dir  # Set default directory for the log file
    )
    return file_path

def main():
    # Prompt the user to browse for the log file
    log_file_path = browse_log_file()
    
    # Check if the log file was selected
    if not log_file_path:
        logging.error("Log file not selected. Exiting.")
        return  # Exit if the user doesn't select a file
    
    # Display the selected log file path and fixed model path
    logging.info(f"Log file selected: {log_file_path}")
    logging.info(f"Using fixed model file: {MODEL_PATH}")

    # Set up a basic stop event for thread control
    stop_event = Event()

    # Call the prediction process without log_signal and resource_signal
    logging.info("Starting the prediction process...")
    try:
        # Process the log file using the fixed model file
        max_memory, max_cpu = process_log_in_intervals(
            file_path=log_file_path,
            model_path=MODEL_PATH,
            stop_event=stop_event,  # Pass the stop_event to manage thread interruptions
            # interval_minutes=5,
            # delay_seconds=5,
            # label_mapping_choice=3
        )

        logging.info(f"Processing complete.\nMax memory usage: {max_memory:.2f} MB\nMax CPU usage: {max_cpu:.2f}%")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()


# import logging
# import tkinter as tk
# from tkinter import filedialog
# from demo_backend import process_log_in_intervals  # Assuming the process_log_in_intervals function is defined in demo_backend.py
# from threading import Event

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def browse_file(file_type):
#     """Allow the user to browse and select a file, with a default directory for each type."""
#     root = tk.Tk()
#     root.withdraw()  # Hide the root window to avoid displaying the tkinter main window
    
#     # Default directories
#     default_log_dir = "C:/Users/USER/Documents/Party/01 - PSU/Project II/Data/"
#     default_model_dir = "C:/Users/USER/Documents/Party/01 - PSU/Project II/Models/"
    
#     if file_type == "log":
#         # Ask user to select the log file, starting in the default directory for log files
#         file_path = filedialog.askopenfilename(
#             title="Select Log File", 
#             filetypes=(("All files", "*.*"), ("Text files", "*.txt")),
#             initialdir=default_log_dir  # Set default directory for the log file
#         )
#     elif file_type == "model":
#         # Ask user to select the model file, starting in the default directory for model files
#         file_path = filedialog.askopenfilename(
#             title="Select Model File", 
#             filetypes=(("Joblib files", "*.joblib"), ("All files", "*.*")),
#             initialdir=default_model_dir  # Set default directory for the model file
#         )
    
#     return file_path

# def main():
#     # Prompt the user to browse for the log and model files
#     log_file_path = browse_file("log")
#     model_file_path = browse_file("model")
    
#     # Check if files were selected
#     if not log_file_path or not model_file_path:
#         logging.error("Log file or model file not selected. Exiting.")
#         return  # Exit if the user doesn't select both files
    
#     # Display the selected file paths
#     logging.info(f"Log file selected: {log_file_path}")
#     logging.info(f"Model file selected: {model_file_path}")

#     # Set up a basic stop event for thread control
#     stop_event = Event()

#     # Call the prediction process without log_signal and resource_signal
#     logging.info("Starting the prediction process...")
#     try:
#         # Process the log file and model file
#         max_memory, max_cpu = process_log_in_intervals(
#             file_path=log_file_path,
#             model_path=model_file_path,
#             stop_event=stop_event,  # Pass the stop_event to manage thread interruptions
#             # interval_minutes=5,
#             # delay_seconds=5,
#             # label_mapping_choice=3
#         )

#         logging.info(f"Processing complete.\nMax memory usage: {max_memory:.2f} MB\nMax CPU usage: {max_cpu:.2f}%")

#     except Exception as e:
#         logging.error(f"An error occurred during processing: {e}")

# if __name__ == "__main__":
#     main()

