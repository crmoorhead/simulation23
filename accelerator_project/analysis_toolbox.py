from contextlib import contextmanager
import io, sys, os
from datetime import datetime
from pathlib import Path

def get_latest_log_number(log_dir):
    # resolve path to correct absolute path
    log_dir_path = Path(log_dir).resolve().parent / f"{Path(log_dir).name}"
    if not log_dir_path.exists():
        log_dir_path.mkdir(parents=True)
        return 0
    # Use str.split("_") to extract the digits after the underscore
    log_numbers = [int(x.name.split("_")[1][:-4]) for x in log_dir_path.glob("run_*.out")]
    return max(log_numbers, default=0)

@contextmanager
def log_terminal_output(log_dir="."):
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_number = get_latest_log_number(log_dir) + 1
    print(log_number)
    log_file_path = log_dir_path / f"run_{log_number:03}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"{log_file_path}.out", 'w') as log_file:
        log_file.write(f"{timestamp}\n\n")
    org_stdout = sys.stdout
    org_stderr = sys.stderr
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    # Open the output and error files for writing
    out_file = open(f"{log_file_path}.out", 'a')
    err_file = open(f"{log_file_path}.err", 'a')
    try:
        sys.stdout = Tee(sys.stdout, captured_stdout, out_file)
        sys.stderr = Tee(sys.stderr, captured_stderr, err_file)
        yield
    finally:
        # Reset the output streams to their original values
        sys.stdout = org_stdout
        sys.stderr = org_stderr
        # Get the output and error messages from the StringIO objects
        out = captured_stdout.getvalue()
        err = captured_stderr.getvalue()
        # Write the output and error messages to the respective files
        out_file.write(out)
        err_file.write(err)
        # Close the output and error files
        out_file.close()
        err_file.close()
        # If the error file is empty, delete it
        if Path(f"{log_file_path}.err").stat().st_size == 0:
            os.remove(f"{log_file_path}.err")
        # Print the log file path
        print(f"Log file saved to: {log_file_path}")
        # Print the output and error messages to the console
        if out:
            print(out)
        if err:
            print(err)

# Tee is designed to duplicate an input stream and write the duplicated data
# to multiple output streams.

# The class takes three file objects as input to its constructor (orig, captured, and log).
# These represent the original input stream, a stream that will capture the duplicated data,
# and a log file where the data will be written.

# When the write method of the Tee class is called, it writes data to all three of the file objects.
# Similarly, the flush method flushes all three files.

class Tee:
    def __init__(self, orig, captured, log):
        self.orig = orig
        self.captured = captured
        self.log = log

    def write(self, data):
        self.orig.write(data)
        self.captured.write(data)
        self.log.write(data)

    def flush(self):
        self.orig.flush()
        self.captured.flush()
        self.log.flush()

# Function to filter a file or a list of files. The output is written to a new file with the same name as the original
# file, but with "_filtered" appended to the name. The contents of the new file are the lines from the original file
# that contain any of the sequences in the sequences list.

def filter_file(files, sequences):
    if isinstance(files, str):
        # If files is a string, convert it to a list with one item
        files = [Path(files)]
    else:
        # Otherwise, assume that files is already a list of strings
        files = [Path(file) for file in files]
    for file in files:
        filename = file.name
        path = file.parent
        with file.open("r") as f:
            lines = f.readlines()
        with (path / f"{filename.split('.')[0]}_filtered.txt").open("w") as f:
            for line in lines:
                # Checking if any of the sequences are present in the line (ignoring case)
                if any(seq.lower() in line.lower() for seq in sequences):
                    # Writing the line to the filtered file
                    f.write(line)

def filter_dir(dir, sequences):
    # Getting the path of the directory
    path = Path(dir)
    # Getting a list of all the files in the directory
    files = path.glob("run_*.log")
    # Filtering the files
    filter_file(files, sequences)
