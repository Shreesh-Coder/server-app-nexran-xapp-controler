import subprocess
import os


def get_nexran_ip():
    try:
        # Run the shell command to get the NEXRAN IP address
        result = subprocess.run(
            "sudo kubectl get svc -n ricxapp --field-selector metadata.name=service-ricxapp-nexran-rmr -o jsonpath='{.items[0].spec.clusterIP}'", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        # Check if the command executed successfully
        if result.returncode != 0:
            print(f"Error executing kubectl command: {result.stderr}")
            return None
        
        # Extract the NEXRAN IP address from the command output
        nexran_ip = result.stdout.strip()
        if nexran_ip:
            # Set the NEXRAN_XAPP environment variable in the Python app
            os.environ['NEXRAN_XAPP'] = nexran_ip
            print(f"NEXRAN_XAPP={nexran_ip}")
            return nexran_ip
        else:
            print("Failed to retrieve NEXRAN IP.")
            return None
        
    except Exception as e:
    # Handle exception
        print(f"An error occurred: {e}")