import os
import logging

def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized file & console logger."""
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger
