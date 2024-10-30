import pandas as pd
import numpy as np
import joblib
import psutil  # Library for monitoring system resources
import time

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

# Function to monitor and print resource usage for each batch
def measure_resource_usage(initial_memory, initial_cpu):
    process = psutil.Process()
    current_memory = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    current_cpu = psutil.cpu_percent(interval=1)  # CPU usage in percentage

    # Calculate the difference (resource used by the current batch)
    memory_used = current_memory - initial_memory
    cpu_used = current_cpu - initial_cpu

    return memory_used, cpu_used

def process_log_in_intervals(file_path, model_path, interval_minutes=5, batch_size=5):
    try:
        # Load the model and signature list
        model_data = joblib.load(model_path)
        clf = model_data['model']
        signature_list = model_data['signature_list']

        # Define time intervals in seconds
        interval_duration = pd.Timedelta(minutes=interval_minutes)

        # Chunk size to read the file in batches
        chunk_size = 5000  # Adjust based on your system's RAM capacity

        # Read the log data in chunks
        df_iterator = pd.read_csv(file_path, delim_whitespace=True, header=None, na_filter=False,
                                  on_bad_lines='skip', skiprows=8, low_memory=False, chunksize=chunk_size)

        # Initialize variables to track current interval
        current_time = None
        next_time = None
        interval_count = 1

        # Initialize variables for batch processing
        interval_buffer = []
        time_buffer = []
        batch_counter = 0

        # Label mapping
        label_mapping = {
            0: 'Attack',
            1: 'C&C',
            2: 'C&C-HeartBeat',
            3: 'C&C-Torii',
            4: 'DDoS',
            5: 'Okiru',
            6: 'PortScan',
            7: 'PortScan-Attack'
        }

        # Measure the initial resource usage before starting processing
        print("Measuring Initial Resource Usage...")
        process = psutil.Process()                                                                                         # Get the current process
        initial_memory_usage = process.memory_info().rss / (1024 * 1024)                                                   # Initial memory usage in MB
        # initial_memory_usage = process.memory_info().rss / 1024                                                          # Initial memory usage in KB
        initial_cpu_usage = psutil.cpu_percent(interval=1)                                                                 # Initial CPU usage over 1 second
        print(f"Initial Memory Usage: {initial_memory_usage:.2f} MB, Initial CPU Usage: {initial_cpu_usage:.2f}%\n")

        start_time = time.time()  # Start time to measure execution duration

        for chunk in df_iterator:
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
                # Filter data for the current interval
                interval_df = chunk[(chunk[0] >= current_time) & (chunk[0] < next_time)]

                if not interval_df.empty:
                    # Generate signatures from the filtered logs
                    signatures = generate_signatures_from_log(interval_df)

                    # Prepare the feature matrix (X_matrix) for prediction
                    X_matrix = np.zeros((len(signatures), len(signature_list)), dtype=int)

                    # Map signatures to the matrix
                    for i, (sig, frequency) in enumerate(signatures):
                        if sig in signature_list:
                            sig_idx = signature_list.index(sig)
                            # Update the matrix for this signature
                            X_matrix[i, sig_idx] = frequency  # Use i to index X_matrix

                    # Add to interval buffer
                    interval_buffer.append(X_matrix)
                    time_buffer.append(current_time)
                    batch_counter += 1

                # If batch size is reached, make predictions for all intervals in the batch
                if batch_counter >= batch_size:
                    for X_mat, interval_time in zip(interval_buffer, time_buffer):
                        # Make predictions for all signatures in the current batch of intervals
                        predictions = clf.predict(X_mat)

                        # Convert numerical predictions to corresponding labels
                        predicted_labels = [label_mapping[pred] for pred in predictions]

                        # Print the unique predicted labels for the interval
                        unique_predicted_labels = list(set(predicted_labels))
                        prediction_counts = {label: predicted_labels.count(label) for label in unique_predicted_labels}
                        print(f"Interval {interval_count}: {interval_time}")
                        print(f"\tPredictions: {prediction_counts}")

                        interval_count += 1

                    # Clear buffers after processing batch
                    interval_buffer = []
                    time_buffer = []
                    batch_counter = 0

                    # Measure resource usage for the current batch and calculate the difference
                    memory_used, cpu_used = measure_resource_usage(initial_memory_usage, initial_cpu_usage)
                    # print(f"\nBatch {interval_count // batch_size} Resource Usage:")
                    print(f"------- Memory Used: {memory_used:.2f} MB\n------- CPU Used: {cpu_used:.2f}%\n")

                # Move to the next interval
                current_time = next_time
                next_time = current_time + interval_duration

        # Process any remaining intervals in the buffer
        if interval_buffer:
            for X_mat, interval_time in zip(interval_buffer, time_buffer):
                predictions = clf.predict(X_mat)
                predicted_labels = [label_mapping[pred] for pred in predictions]
                unique_predicted_labels = list(set(predicted_labels))
                prediction_counts = {label: predicted_labels.count(label) for label in unique_predicted_labels}
                print(f"Interval {interval_count}: {interval_time}")
                print(f"\tPredictions: {prediction_counts}")
                interval_count += 1

        # Measure final resource usage
        print("\nFinal Resource Usage:")
        memory_used, cpu_used = measure_resource_usage(initial_memory_usage, initial_cpu_usage)
        print(f"\tMemory Used: {memory_used:.2f} MB, CPU Used: {cpu_used:.2f}%\n")

        # Execution time
        end_time = time.time()
        print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

    except FileNotFoundError:
        print(f"File not found: {file_path} or {model_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Log Data to choose
# log_file_path = "C:/Users/Natty PC/Documents/Party/Project II/Data/Capture-17-1.labeled"
# log_file_path = "C:/Users/Natty PC/Documents/Party/Project II/Data/Capture-3-1.labeled"
log_file_path = "C:/Users/Natty PC/Documents/Party/Project II/Data/Capture-34-1.labeled"
# log_file_path = "C:/Users/Natty PC/Documents/Party/Project II/Data/Capture-36-1.labeled"
# log_file_path = "C:/Users/Natty PC/Documents/Party/Project II/Data/Capture-60-1.labeled"
# log_file_path = "C:/Users/Natty PC/Documents/Party/Project II/Data/corrected_traffic_dataset.labeled"

# Trained Model to choose
model_path = "C:/Users/Natty PC/Documents/Party/Project II/Models/Model - DCT-mali.joblib"
# model_path = "C:/Users/Natty PC/Documents/Party/Project II/Models/Model - SVM-mali.joblib"
# model_path = "C:/Users/Natty PC/Documents/Party/Project II/Models/Model - RF-mali.joblib"
# model_path = "C:/Users/Natty PC/Documents/Party/Project II/Models/Model - AdaBoost-mali.joblib"

process_log_in_intervals(log_file_path, model_path, interval_minutes=5)
