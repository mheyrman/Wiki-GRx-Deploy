from datetime import datetime
import logging
import sys
import os
import signal

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

class RunLogger:
    def __init__(self, log_dir='logs', run_name=None):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        if run_name is None:
            run_name = "n1_deploy"
            self.log_file = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
        else:
            self.log_file = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.log_file = os.path.join(self.log_dir, self.log_file)

        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Attach handlers (avoid duplicates)
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self._tee_file = open(self.log_file, 'a', buffering=1)
        sys.stdout = Tee(sys.__stdout__, self._tee_file)
        sys.stderr = Tee(sys.__stderr__, self._tee_file)

        self.logger.info(f"Logger initialized for run: {run_name}")

        def _sigint_handler(signum, frame):
            print("\n[RunLogger] Caught CTRL+C KeyboardInterrupt, closing logger...")
            self.close()
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, _sigint_handler)

    def log_step(self, step, **data):
        """
        Log data for a single time step.

        Args:
            step (int): Current time step or iteration.
            **data: Arbitrary key-value pairs to log (e.g., loss=0.23, accuracy=0.89)
        """
        msg = f"Step {step} | " + " | ".join(f"{k}={v}" for k, v in data.items())
        self.logger.info(msg)
    
    def log_message(self, message, level="info"):
        """
        Log a custom message at the specified level.
        """
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
        else:
            self.logger.info(message)

    def close(self):
        """
        Close the logger and restore stdout/stderr.
        """
        self.logger.info(f"[RunLogger] Closing logger... {self.log_file}")

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        if hasattr(self, '_tee_file'):
            self._tee_file.close()

        for handler in self.logger.handlers:
            handler.close()

        self.logger.handlers.clear()

        self.logger.info(f"[RunLogger] Logger closed: {self.log_file}")