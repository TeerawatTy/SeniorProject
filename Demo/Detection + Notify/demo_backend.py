# demo_backend.py

import pandas as pd
import numpy as np
import joblib
import psutil
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import requests
import pdb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LINE Notify details
LINE_TOKEN = "TjZoKkib0jdDIfh2omhORZ5lKwhC1tCK2TsYdAzWJ1j"
LINE_API_URL = "https://notify-api.line.me/api/notify"

# Function to generate signatures from a DataFrame
def generate_signatures_from_log(df):
    signature_counts = {}
    
    for _, row in df.iterrows():
        id_resp_p = row[5]
        proto = row[6]
        service = row[7]
        orig_bytes = row[9]
        resp_bytes = row[10]
        conn_state = row[11]

        # Clean up service and byte values
        service = service if service not in ['0', '-', ''] else ''
        orig_bytes = orig_bytes if orig_bytes not in ['-', '0'] else '0'
        resp_bytes = resp_bytes if resp_bytes not in ['-', '0'] else '0'

        # Create the signature
        signature = f"{id_resp_p}{proto}{service}{orig_bytes}{resp_bytes}{conn_state}"

        # Count occurrences of each signature
        signature_counts[signature] = signature_counts.get(signature, 0) + 1

    output_list = [[signature, frequency] for signature, frequency in signature_counts.items()]
    return output_list

# Function to send Line Notification
# Function to send Line Notification
def send_line_notification(attack_summary, interval_id):
    """Send a Line Notify message for detected attacks."""
    if not attack_summary:
        logging.info("No attacks detected, no notification sent.")
        return  # Nothing to notify

    try:
        # Construct the notification message
        message = "\n".join(
            [
                # f"\n»»»» Attack Detected ««««\n"
                f"\n⚠️⚠️⚠️ Attack Detected ⚠️⚠️⚠️\n"
                f"🕒 Interval ID: {interval_id}\n"  # Include interval_id in the notification
                f"⚔️ Type: {attack['label']}\n"
                f"🤖 From IP Address: {attack['source_ip']}\n"
                f"🗝️ Port: {attack['receive_port']}\n"
                f"🔄️ Frequency: {attack['frequency']} times\n"
                f"📅 Timestamp: {attack['timestamp']}\n"
                for attack in attack_summary
                if attack.get('source_ip') not in ['Unknown', 'Invalid_IP', '0'] and attack.get('receive_port') not in ['Unknown', 'Invalid_Port', '0']
            ]
        )

        if not message:
            logging.info("No valid attacks in the summary to notify about.")
            return
        
        # Send message using Line Notify API
        headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
        data = {"message": message}
        response = requests.post(LINE_API_URL, headers=headers, data=data)

        # Check response status
        if response.status_code == 200:
            logging.info("Notification sent successfully.")
        else:
            logging.error(f"Failed to send notification: {response.status_code} - {response.text}")
    
    except Exception as e:
        logging.error(f"Error while sending notification: {str(e)}")


# Function to measure resource usage
def measure_resource_usage():
    process = psutil.Process()
    current_memory = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    current_cpu = psutil.cpu_percent(interval=1)  # CPU usage in percentage
    return current_memory, current_cpu

# Function to dynamically allocate chunk size based on available memory
def get_dynamic_chunk_size():
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    chunk_size = int(min(5000, max(1000, available_memory // 10)))  # scale with available memory
    logging.info(f"Dynamically set chunk size: {chunk_size}")
    return chunk_size

def process_batch(interval_buffer, time_buffer, clf, signature_list, label_mapping, batch_id, log_signal, resource_signal, stop_event, interval_id):
    logging.info(f"Processing batch {batch_id}...")

    attack_summary = []  # List to store attack summaries
    seen_attacks = set()  # Set to track already seen (source_ip, receive_port) combinations

    # Process each interval in the batch
    for idx, interval_df in enumerate(interval_buffer):
        # Generate signatures for the current interval
        signatures = generate_signatures_from_log(interval_df)

        # Prepare feature matrix for model prediction
        X_matrix = np.zeros((len(signatures), len(signature_list)), dtype=int)
        for i, (sig, frequency) in enumerate(signatures):
            if sig in signature_list:
                sig_idx = signature_list.index(sig)
                X_matrix[i, sig_idx] = frequency

        # Predict attack labels using the model
        predictions = clf.predict(X_matrix)

        # Collect detected anomalies (non-'Benign')
        for i, prediction in enumerate(predictions):
            if label_mapping[prediction] != 'Benign':
                source_ip = interval_df.iloc[i][2]  # Changed to column 2 as requested
                receive_port = interval_df.iloc[i][5]
                timestamp = interval_df.iloc[i][0]  # Extract timestamp from column 0

                # Check if this attack (IP, port) combination has already been seen
                if (source_ip, receive_port) not in seen_attacks:
                    seen_attacks.add((source_ip, receive_port))
                    attack_summary.append({
                        'label': label_mapping[prediction],
                        'source_ip': source_ip,
                        'receive_port': receive_port,
                        'frequency': signatures[i][1],
                        'timestamp': timestamp,  # Add the timestamp to the attack summary
                        'interval_id': interval_id  # Include the interval ID
                    })

        # Immediately notify if any anomaly (non-'Benign') is detected
        if attack_summary:
            send_line_notification(attack_summary, interval_id)

        # Clear attack_summary for the next interval
        attack_summary = []

    # Measure resource usage for the batch
    current_memory_usage, current_cpu_usage = measure_resource_usage()
    logging.info(f"Batch {batch_id} processed. Memory usage: {current_memory_usage:.2f} MB, CPU usage: {current_cpu_usage:.2f}%")

    # Emit resource usage if signal is provided
    if resource_signal:
        resource_signal.emit(f"Resource Usage for Batch {batch_id}: Memory Used: {current_memory_usage:.2f} MB, CPU Used: {current_cpu_usage:.2f}%")


def process_log_in_intervals(file_path, model_path, interval_minutes=5, log_signal=None, resource_signal=None, intervals_per_batch=5, stop_event=None, label_mapping_choice=1, delay_seconds=2):
    max_memory_usage = 0  # Track max memory usage
    max_cpu_usage = 0      # Track max CPU usage
    
    try:
        # Load the model and signature list
        logging.info("Loading model data...")
        model_data = joblib.load(model_path)
        clf = model_data['model']
        signature_list = model_data['signature_list']
        
        # Debug: Check the loaded model and signature list
        # pdb.set_trace()  # Check model and signature list

        # Use 1st or 2nd Label Mapping based on the choice

        if label_mapping_choice == 1:
            label_mapping = {
                0: 'Benign',
                1: 'Benign',
                2: 'C&C',
                3: 'C&C-HeartBeat',
                4: 'C&C-Torii',
                5: 'DDoS',
                6: 'Okiru',
                7: 'PortScan'
            }
        
        elif label_mapping_choice == 2:
            label_mapping = {
                0: 'Benign',
                1: 'C&C',
                2: 'C&C-HeartBeat',
                3: 'C&C-Torii',
                4: 'DDoS',
                5: 'Okiru',
                6: 'PortScan',
                7: 'PortScan-Attack'
            }
        elif label_mapping_choice == 3:
            label_mapping = {
                0: 'Attack',
                1: 'C&C',
                2: 'Benign',
                3: 'C&C-HeartBeat',
                4: 'C&C-Torii',
                5: 'DDoS',
                6: 'Okiru',
                7: 'PortScan'
            }

        # Debug: Check the label mapping
        # pdb.set_trace()  # Check label mapping
        
        # Define time intervals in seconds
        interval_duration = pd.Timedelta(minutes=interval_minutes)

        # Dynamically set chunk size based on system resources
        chunk_size = get_dynamic_chunk_size()
        logging.info(f"Dynamically set chunk size: {chunk_size}")

        # Read the log data in chunks
        df_iterator = pd.read_csv(file_path, delim_whitespace=True, header=None, na_filter=False,
                                  on_bad_lines='skip', skiprows=8, low_memory=False, chunksize=chunk_size)

        # Initialize variables for batch processing
        current_time = None
        next_time = None
        interval_count = 1  # This is your unique interval ID
        interval_buffer = []
        time_buffer = []
        interval_counter = 0  # Keep track of the number of intervals in a batch

        # Use ThreadPoolExecutor to process batches in parallel
        with ThreadPoolExecutor() as executor:
            for chunk in df_iterator:
                if stop_event.is_set():  # Check if the stop event is set
                    logging.info("Stopping processing of log due to user request.")
                    break  # Exit the loop if stopped

                # Debug: Check the chunk before preprocessing
                logging.info(f"Processing chunk with {len(chunk)} rows")
                if chunk.empty:
                    logging.warning("This chunk is empty!")
                # pdb.set_trace()  # Check chunk before processing

                # Preprocess the chunk
                chunk[0] = pd.to_numeric(chunk[0], errors='coerce')
                chunk.dropna(subset=[0, 3, 5, 6, 14, 16, 17], inplace=True)
                chunk[0] = pd.to_datetime(chunk[0], errors='coerce', unit='s')  # Convert timestamp to datetime
                chunk.dropna(subset=[0], inplace=True)

                # Debug: Check the chunk after preprocessing
                # pdb.set_trace()  # Check chunk after processing

                if current_time is None:
                    current_time = chunk[0].min()
                    next_time = current_time + interval_duration

                # Process the chunk in intervals
                while current_time < chunk[0].max():
                    interval_df = chunk[(chunk[0] >= current_time) & (chunk[0] < next_time)]

                    # Debug: Check the interval data frame
                    logging.debug(f"Processing interval: {current_time} - {next_time}")
                    # pdb.set_trace()  # Check interval_df for each interval

                    if not interval_df.empty:
                        # Generate signatures
                        signatures = generate_signatures_from_log(interval_df)

                        # Debug: Check generated signatures
                        logging.debug("Generated signatures:")
                        # pdb.set_trace()  # Check generated signatures

                        # Prepare feature matrix
                        X_matrix = np.zeros((len(signatures), len(signature_list)), dtype=int)
                        for i, (sig, frequency) in enumerate(signatures):
                            if sig in signature_list:
                                sig_idx = signature_list.index(sig)
                                X_matrix[i, sig_idx] = frequency

                        # Debug: Check the feature matrix
                        logging.debug(f"Feature matrix:\n{X_matrix}")
                        # pdb.set_trace()  # Check X_matrix

                        # Predict attack labels using the model
                        predictions = clf.predict(X_matrix)

                        # Debug: Check predictions
                        logging.debug(f"Predictions: {predictions}")
                        # pdb.set_trace()  # Check predictions

                        # Log predictions to a file
                        # for i, prediction in enumerate(predictions):
                        #     predicted_label = label_mapping.get(prediction, 'Unknown')
                        #     log_message = f"Interval ID: {interval_count}, Prediction: {predicted_label}, Signature Index: {i}"
                        #     logging.info(log_message)

                        # Check if there is any anomaly (non-'benign') to notify
                        attack_summary = []
                        for i, prediction in enumerate(predictions):
                            if label_mapping[prediction] != 'Benign':
                                attack_summary.append({
                                    'label': label_mapping[prediction],
                                    'source_ip': interval_df.iloc[i][2],  # Source IP
                                    'receive_port': interval_df.iloc[i][5],  # Receive Port
                                    'frequency': signatures[i][1],
                                    'timestamp': interval_df.iloc[i][0],
                                })
                                

                        # Send notification if an attack is detected
                        if attack_summary:
                            interval_id = interval_count  # Define the interval_id based on the interval_count
                            send_line_notification(attack_summary, interval_id)  # Pass interval_id

                    # Add delay here to simulate slower processing (delay_seconds is in seconds)
                    time.sleep(delay_seconds)

                    # Debug: Check interval buffers and counter before processing batches
                    logging.debug(f"Interval buffer size: {len(interval_buffer)}; Time buffer size: {len(time_buffer)}; Interval counter: {interval_counter}")
                    # pdb.set_trace()  # Check interval buffer status

                    # Process a batch of intervals
                    if interval_counter >= intervals_per_batch:
                        batch_id = interval_count // intervals_per_batch

                        # Log predictions for each interval in the batch
                        executor.submit(process_batch, interval_buffer, time_buffer, clf, signature_list, label_mapping, batch_id, log_signal, resource_signal, stop_event, interval_counter)

                        # Measure resource usage
                        current_memory_usage, current_cpu_usage = measure_resource_usage()
                        max_memory_usage = max(max_memory_usage, current_memory_usage)
                        max_cpu_usage = max(max_cpu_usage, current_cpu_usage)

                        if resource_signal:
                            resource_signal.emit(f"Resource Usage: Memory Used: {current_memory_usage:.2f} MB, CPU Used: {current_cpu_usage:.2f}%")

                        # Clear buffers and reset interval counter
                        interval_buffer = []
                        time_buffer = []
                        interval_counter = 0

                    # Move to the next interval
                    current_time = next_time
                    next_time += interval_duration
                    interval_count += 1  # Increment the interval count to ensure unique IDs

        logging.info(f"Processing complete. Maximum memory usage: {max_memory_usage:.2f} MB, Maximum CPU usage: {max_cpu_usage:.2f}%")
    except Exception as e:
        logging.exception(f"An error occurred while processing log in intervals: {e}")
    
    return max_memory_usage, max_cpu_usage


# def process_log_in_intervals(file_path, model_path, interval_minutes=5, log_signal=None, resource_signal=None, intervals_per_batch=5, stop_event=None, label_mapping_choice=1, delay_seconds=2):
#     max_memory_usage = 0  # Track max memory usage
#     max_cpu_usage = 0      # Track max CPU usage
    
#     try:
#         # Load the model and signature list
#         model_data = joblib.load(model_path)
#         clf = model_data['model']
#         signature_list = model_data['signature_list']

#         # Use 1st or 2nd Label Mapping based on the choice
#         if label_mapping_choice == 1:
#             label_mapping = {
#                 0: 'Benign',
#                 1: 'C&C',
#                 2: 'C&C-HeartBeat',
#                 3: 'C&C-Torii',
#                 4: 'DDoS',
#                 5: 'Okiru',
#                 6: 'PortScan',
#                 7: 'PortScan-Attack'
#             }
#         else:
#             label_mapping = {
#                 0: 'Benign',
#                 1: 'Attack',
#                 2: 'C&C',
#                 3: 'C&C-HeartBeat',
#                 4: 'C&C-Torii',
#                 5: 'DDoS',
#                 6: 'Okiru',
#                 7: 'PortScan'
#             }

#         # Define time intervals in seconds
#         interval_duration = pd.Timedelta(minutes=interval_minutes)

#         # Dynamically set chunk size based on system resources
#         chunk_size = get_dynamic_chunk_size()

#         # Read the log data in chunks
#         df_iterator = pd.read_csv(file_path, delim_whitespace=True, header=None, na_filter=False,
#                                   on_bad_lines='skip', skiprows=8, low_memory=False, chunksize=chunk_size)

#         # Initialize variables for batch processing
#         current_time = None
#         next_time = None
#         interval_count = 1  # This is your unique interval ID
#         interval_buffer = []
#         time_buffer = []
#         interval_counter = 0  # Keep track of the number of intervals in a batch

#         # Use ThreadPoolExecutor to process batches in parallel
#         with ThreadPoolExecutor() as executor:
#             for chunk in df_iterator:
#                 if stop_event.is_set():  # Check if the stop event is set
#                     logging.info("Stopping processing of log due to user request.")
#                     break  # Exit the loop if stopped

#                 # Preprocess the chunk
#                 chunk[0] = pd.to_numeric(chunk[0], errors='coerce')
#                 chunk.dropna(subset=[0, 3, 5, 6, 14, 16, 17], inplace=True)
#                 chunk[0] = pd.to_datetime(chunk[0], errors='coerce', unit='s')  # Convert timestamp to datetime
#                 chunk.dropna(subset=[0], inplace=True)

#                 if current_time is None:
#                     current_time = chunk[0].min()
#                     next_time = current_time + interval_duration

#                 # Process the chunk in intervals
#                 while current_time < chunk[0].max():
#                     interval_df = chunk[(chunk[0] >= current_time) & (chunk[0] < next_time)]

#                     if not interval_df.empty:
#                         # Generate signatures
#                         signatures = generate_signatures_from_log(interval_df)

#                         # Prepare feature matrix
#                         X_matrix = np.zeros((len(signatures), len(signature_list)), dtype=int)
#                         for i, (sig, frequency) in enumerate(signatures):
#                             if sig in signature_list:
#                                 sig_idx = signature_list.index(sig)
#                                 X_matrix[i, sig_idx] = frequency

#                         # Predict attack labels using the model
#                         predictions = clf.predict(X_matrix)

#                         # Check if there is any anomaly (non-'benign') to notify
#                         attack_summary = []
#                         for i, prediction in enumerate(predictions):
#                             if label_mapping[prediction] != 'Benign':
#                                 attack_summary.append({
#                                     'label': label_mapping[prediction],
#                                     'source_ip': interval_df.iloc[i][2],  # Source IP
#                                     'receive_port': interval_df.iloc[i][5],  # Receive Port
#                                     'frequency': signatures[i][1],
#                                     'timestamp': interval_df.iloc[i][0],
#                                 })

#                         # Send notification if an attack is detected
#                         if attack_summary:
#                             interval_id = interval_count  # Define the interval_id based on the interval_count
#                             send_line_notification(attack_summary, interval_id)  # Pass interval_id

#                     # Add delay here to simulate slower processing (delay_seconds is in seconds)
#                     time.sleep(delay_seconds)

#                     # Process a batch of intervals
#                     if interval_counter >= intervals_per_batch:
#                         batch_id = interval_count // intervals_per_batch

#                         # Log predictions for each interval in the batch
#                         executor.submit(process_batch, interval_buffer, time_buffer, clf, signature_list, label_mapping, batch_id, log_signal, resource_signal, stop_event, interval_counter)

#                         # Measure resource usage
#                         current_memory_usage, current_cpu_usage = measure_resource_usage()
#                         max_memory_usage = max(max_memory_usage, current_memory_usage)
#                         max_cpu_usage = max(max_cpu_usage, current_cpu_usage)

#                         if resource_signal:
#                             resource_signal.emit(f"Resource Usage: Memory Used: {current_memory_usage:.2f} MB, CPU Used: {current_cpu_usage:.2f}%")

#                         # Clear buffers and reset interval counter
#                         interval_buffer = []
#                         time_buffer = []
#                         interval_counter = 0

#                     # Move to the next interval
#                     current_time = next_time
#                     next_time += interval_duration
#                     interval_count += 1  # Increment the interval count to ensure unique IDs

#         logging.info(f"Processing complete. Maximum memory usage: {max_memory_usage:.2f} MB, Maximum CPU usage: {max_cpu_usage:.2f}%")
#     except Exception as e:
#         logging.exception(f"An error occurred while processing log in intervals: {e}")
    
#     return max_memory_usage, max_cpu_usage

# # Main entry
# if __name__ == "__main__":
#     log_file_path = "C:/Users/USER/Documents/Party/01 - PSU/Project II/Data/CTU-Mixed-5.labeled"
#     model_file_path = "C:/Users/USER/Documents/Party/01 - PSU/Project II/Models/Model - AdaBoost-mali.joblib"
    
#     stop_event = Event()
#     process_log_in_intervals(log_file_path, model_file_path, stop_event=stop_event, delay_seconds=5)
