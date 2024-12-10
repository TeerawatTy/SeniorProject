import logging
from backend import process_log_in_intervals  # Assuming the process_log_in_intervals function is defined in demo_backend.py
from threading import Event

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Static paths for model and log file
MODEL_PATH = "/home/party/SeniorProject/Models/Model - DCT.joblib"
LOG_PATH = "/home/party/SeniorProject/Data/Capture-34-1.labeled"

def main():
    # Use the static LOG_PATH and MODEL_PATH directly
    log_file_path = LOG_PATH
    model_file_path = MODEL_PATH
    
    # Check if the log file exists at the static path
    if not log_file_path:
        logging.error("Log file not found. Exiting.")
        return  # Exit if the log file is not found
    
    if not model_file_path:
        logging.error("Model file not found. Exiting.")
        return  # Exit if the model file is not found

    # Display the selected log file path and fixed model path
    logging.info(f"Log file selected: {log_file_path}")
    logging.info(f"Using fixed model file: {model_file_path}")

    # Set up a basic stop event for thread control
    stop_event = Event()

    # Call the prediction process
    logging.info("Starting the prediction process...")
    try:
        # Process the log file using the fixed model file
        max_memory, max_cpu = process_log_in_intervals(
            file_path=log_file_path,
            model_path=model_file_path,
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
