import psutil
import os
import signal
import time

def find_python_pids():
    """
    Finds PIDs of all running processes that are instances of 'python' or 'python.exe'.
    """
    python_pids = []
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if the process name or a command line argument indicates it's a Python process
            process_name = proc.info['name'].lower()
            if 'python' in process_name or 'python.exe' in process_name:
                # Exclude the current script's PID to only find *other* processes
                if proc.info['pid'] != os.getpid():
                    python_pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Handle potential errors while accessing process info
            continue
    return python_pids


def kill_process_os(pid):
    """Kills a process with a given PID using os.kill()."""
    try:
        # On POSIX systems, signal.SIGKILL is used.
        # On Windows, os.kill() terminates the process immediately.
        os.kill(pid, signal.SIGKILL)
        print(f"Process with PID {pid} has been killed.")
    except OSError as e:
        print(f"Error killing process {pid}: {e}")


if __name__ == "__main__":
    while True:
        pids = find_python_pids()
        if pids:
            print(f"Found PIDs for other running Python processes: {pids}")
        else:
            break

        for pid in pids:
            kill_process_os(pid)
        
        time.sleep(1)

        # You might launch a dummy process in the background to test this script, for example:
        # import subprocess
        # subprocess.Popen(['python', '-c', 'import time; time.sleep(100)'])
