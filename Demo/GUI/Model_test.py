import pandas as pd
import numpy as np
import joblib
import psutil  # Library for monitoring system resources
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event  # To handle thread stopping

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Function to monitor and print resource usage
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

# Function to process a batch and log interval predictions
def process_batch(interval_buffer, time_buffer, clf, signature_list, label_mapping, batch_id, log_signal, resource_signal, stop_event):
    for idx, (X_mat, interval_time) in enumerate(zip(interval_buffer, time_buffer)):
        if stop_event.is_set():  # Check if the stop event is set
            logging.info("Processing stopped.")
            return  # Exit the function if stopped

        # Make predictions for all signatures in the current batch of intervals
        try:
            predictions = clf.predict(X_mat)

            # Convert numerical predictions to corresponding labels
            predicted_labels = [label_mapping[pred] for pred in predictions]

            # Print the unique predicted labels for the interval
            unique_predicted_labels = list(set(predicted_labels))
            prediction_counts = {label: predicted_labels.count(label) for label in unique_predicted_labels}
            
            # Emit results to GUI
            if log_signal:
                log_signal.emit(f"Batch {batch_id} Interval {interval_time}: {prediction_counts}")

        except Exception as e:
            logging.error(f"Error processing batch {batch_id}: {e}")
            if log_signal:
                log_signal.emit(f"Error processing batch {batch_id}: {str(e)}")

# Main processing function
def process_log_in_intervals(file_path, model_path, interval_minutes=5, log_signal=None, resource_signal=None, intervals_per_batch=5, stop_event=None, label_mapping_choice=1):
    max_memory_usage = 0  # Track max memory usage
    max_cpu_usage = 0      # Track max CPU usage
    
    try:
        # Load the model and signature list
        model_data = joblib.load(model_path)
        clf = model_data['model']
        signature_list = model_data['signature_list']

        # Use 1st or 2nd Label Mapping based on the choice
        if label_mapping_choice == 1:
            label_mapping = {
                0: 'Benign',
                # 0: 'Attack',
                1: 'C&C',
                2: 'C&C-HeartBeat',
                3: 'C&C-Torii',
                4: 'DDoS',
                5: 'Okiru',
                6: 'PortScan',
                7: 'PortScan-Attack'
            }
        else:
            label_mapping = {
                0: 'Benign',
                1: 'Attack',
                2: 'C&C',
                3: 'C&C-HeartBeat',
                4: 'C&C-Torii',
                5: 'DDoS',
                6: 'Okiru',
                7: 'PortScan'
            }

        # Define time intervals in seconds
        interval_duration = pd.Timedelta(minutes=interval_minutes)

        # Dynamically set chunk size based on system resources
        chunk_size = get_dynamic_chunk_size()

        # Read the log data in chunks
        df_iterator = pd.read_csv(file_path, delim_whitespace=True, header=None, na_filter=False,
                                  on_bad_lines='skip', skiprows=8, low_memory=False, chunksize=chunk_size)

        # Initialize variables for batch processing
        current_time = None
        next_time = None
        interval_count = 1
        interval_buffer = []
        time_buffer = []
        interval_counter = 0  # Keep track of the number of intervals in a batch

        # Use ThreadPoolExecutor to process batches in parallel
        with ThreadPoolExecutor() as executor:
            for chunk in df_iterator:
                if stop_event.is_set():  # Check if the stop event is set
                    logging.info("Stopping processing of log due to user request.")
                    break  # Exit the loop if stopped

                # Preprocess the chunk
                chunk[0] = pd.to_numeric(chunk[0], errors='coerce')
                chunk.dropna(subset=[0, 3, 5, 6, 14, 16, 17], inplace=True)
                chunk[0] = pd.to_datetime(chunk[0], errors='coerce', unit='s')  # Convert timestamp to datetime
                chunk.dropna(subset=[0], inplace=True)

                if current_time is None:
                    current_time = chunk[0].min()
                    next_time = current_time + interval_duration

                # Process the chunk in intervals
                while current_time < chunk[0].max():
                    interval_df = chunk[(chunk[0] >= current_time) & (chunk[0] < next_time)]

                    if not interval_df.empty:
                        # Generate signatures
                        signatures = generate_signatures_from_log(interval_df)

                        # Prepare feature matrix
                        X_matrix = np.zeros((len(signatures), len(signature_list)), dtype=int)
                        for i, (sig, frequency) in enumerate(signatures):
                            if sig in signature_list:
                                sig_idx = signature_list.index(sig)
                                X_matrix[i, sig_idx] = frequency

                        # Add to interval buffer
                        interval_buffer.append(X_matrix)
                        time_buffer.append(current_time)
                        interval_counter += 1

                    # Process a batch of intervals
                    if interval_counter >= intervals_per_batch:
                        batch_id = interval_count // intervals_per_batch

                        # Log predictions for each interval in the batch
                        executor.submit(process_batch, interval_buffer, time_buffer, clf, signature_list, label_mapping, batch_id, log_signal, resource_signal, stop_event)

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

                    # Move to next interval
                    current_time = next_time
                    next_time += interval_duration

        logging.info(f"Processing complete. Maximum memory usage: {max_memory_usage:.2f} MB, Maximum CPU usage: {max_cpu_usage:.2f}%")
    except Exception as e:
        logging.exception(f"An error occurred while processing log in intervals: {e}")
    
    return max_memory_usage, max_cpu_usage

# def process_log_in_intervals(file_path, model_path, interval_minutes=5, log_signal=None, resource_signal=None, intervals_per_batch=5, stop_event=None):
#     max_memory_usage = 0  # Track max memory usage
#     max_cpu_usage = 0      # Track max CPU usage
    
#     try:
#         # Load the model and signature list
#         model_data = joblib.load(model_path)
#         clf = model_data['model']
#         signature_list = model_data['signature_list']

#         # 1st Label mapping
#         # label_mapping = {
#         #     0: 'Attack',
#         #     1: 'C&C',
#         #     2: 'C&C-HeartBeat',
#         #     3: 'C&C-Torii',
#         #     4: 'DDoS',
#         #     5: 'Okiru',
#         #     6: 'PortScan',
#         #     7: 'PortScan-Attack'
#         # }

#         # 2nd Label Mapping 
#         label_mapping = {
#             0: 'Attack',
#             1: 'Benign',
#             2: 'C&C',
#             3: 'C&C-HeartBeat',
#             4: 'C&C-Torii',
#             5: 'DDoS',
#             6: 'Okiru',
#             7: 'PortScan'
#         }

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
#         interval_count = 1
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

#                         # Add to interval buffer
#                         interval_buffer.append(X_matrix)
#                         time_buffer.append(current_time)
#                         interval_counter += 1

#                     # Process a batch of intervals
#                     if interval_counter >= intervals_per_batch:
#                         batch_id = interval_count // intervals_per_batch

#                         # Log predictions for each interval in the batch
#                         executor.submit(process_batch, interval_buffer, time_buffer, clf, signature_list, label_mapping, batch_id, log_signal, resource_signal, stop_event)

#                         # Measure resource usage
#                         current_memory_usage, current_cpu_usage = measure_resource_usage()
#                         max_memory_usage = max(max_memory_usage, current_memory_usage)
#                         max_cpu_usage = max(max_cpu_usage, current_cpu_usage)

#                         if resource_signal:
#                             resource_signal.emit(f"Resource Usage: Memory Used: {current_memory_usage:.2f} MB, CPU Used: {current_cpu_usage:.2f}%")

#                         # Clear buffers and reset interval counter after batch processing
#                         interval_buffer, time_buffer = [], []
#                         interval_counter = 0
#                         interval_count += 1

#                     current_time = next_time
#                     next_time = current_time + interval_duration

#             # Process any remaining intervals if they're less than the batch size
#             if interval_buffer and not stop_event.is_set():
#                 batch_id = interval_count // intervals_per_batch
#                 process_batch(interval_buffer, time_buffer, clf, signature_list, label_mapping, batch_id, log_signal, resource_signal, stop_event)

#     except FileNotFoundError:
#         logging.error(f"File not found: {file_path} or {model_path}")
#         if log_signal:
#             log_signal.emit(f"File not found: {file_path} or {model_path}")
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         if log_signal:
#             log_signal.emit(f"An error occurred: {str(e)}")
    
#     # Return max memory and CPU usage
#     return max_memory_usage, max_cpu_usage
