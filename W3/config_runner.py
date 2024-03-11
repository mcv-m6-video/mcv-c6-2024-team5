import json
import subprocess

def run_configurations(config_file, python_script):
    with open(config_file, 'r') as f:
        configurations = json.load(f)

    for config in configurations:
        args = []
        for key, value in config.items():
            # For boolean flags, include the flag only if the value is True
            if isinstance(value, bool):
                if value:
                    args.append(key)
            else:
                args.extend([key, str(value)])
        
        # Construct and execute the command
        command = ["python", python_script] + args
        print("Running command:", ' '.join(command))
        subprocess.run(command)

if __name__ == "__main__":
    CONFIG_FILE = "configurations.json"
    PYTHON_SCRIPT = "main.py"  # Ensure this path is correct

    run_configurations(CONFIG_FILE, PYTHON_SCRIPT)
