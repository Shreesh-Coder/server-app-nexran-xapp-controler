import requests

BASE_URL = "http://localhost:your_port/v1"  # Replace 'your_port' with the actual port used by your RestServer.

class NexranClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url

    def get_version(self):
        return requests.get(f"{self.base_url}/version")

    def get_app_config(self):
        return requests.get(f"{self.base_url}/appconfig")

    def put_app_config(self, config_data):
        return requests.put(f"{self.base_url}/appconfig", json=config_data)

    def get_nodebs(self):
        return requests.get(f"{self.base_url}/nodebs")

    def get_nodeb(self, name):
        return requests.get(f"{self.base_url}/nodebs/{name}")

    def post_nodeb(self, nodeb_data):
        return requests.post(f"{self.base_url}/nodebs", json=nodeb_data)

    def put_nodeb(self, name, nodeb_data):
        return requests.put(f"{self.base_url}/nodebs/{name}", json=nodeb_data)

    def delete_nodeb(self, name):
        return requests.delete(f"{self.base_url}/nodebs/{name}")

    def post_nodeb_slice_binding(self, nodeb_name, slice_name):
        return requests.post(f"{self.base_url}/nodebs/{nodeb_name}/slices/{slice_name}")

    def delete_nodeb_slice_binding(self, nodeb_name, slice_name):
        return requests.delete(f"{self.base_url}/nodebs/{nodeb_name}/slices/{slice_name}")

    def get_slices(self):
        return requests.get(f"{self.base_url}/slices")

    def get_slice(self, name):
        return requests.get(f"{self.base_url}/slices/{name}")

    def post_slice(self, slice_data):
        return requests.post(f"{self.base_url}/slices", json=slice_data)

    def put_slice(self, name, slice_data):
        return requests.put(f"{self.base_url}/slices/{name}", json=slice_data)

    def delete_slice(self, name):
        return requests.delete(f"{self.base_url}/slices/{name}")

    def post_slice_ue_binding(self, slice_name, imsi):
        return requests.post(f"{self.base_url}/slices/{slice_name}/ues/{imsi}")

    def delete_slice_ue_binding(self, slice_name, imsi):
        return requests.delete(f"{self.base_url}/slices/{slice_name}/ues/{imsi}")

    def get_ues(self):
        return requests.get(f"{self.base_url}/ues")

    def get_ue(self, imsi):
        return requests.get(f"{self.base_url}/ues/{imsi}")

    def post_ue(self, ue_data):
        return requests.post(f"{self.base_url}/ues", json=ue_data)

    def put_ue(self, imsi, ue_data):
        return requests.put(f"{self.base_url}/ues/{imsi}", json=ue_data)

    def delete_ue(self, imsi):
        return requests.delete(f"{self.base_url}/ues/{imsi}")

# Example Usage:
# nexran_client = NexranClient(base_url="http://localhost:9080/v1")
# response = nexran_client.get_nodebs()
# print(response.json())