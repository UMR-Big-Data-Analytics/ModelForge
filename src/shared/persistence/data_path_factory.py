class DataPathFactory:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def get_raw_data_path(self, device_uuid: str) -> str:
        return f"{self.input_path}/{device_uuid}"

    def get_processed_data_path(self, device_uuid: str) -> str:
        return f"{self.output_path}/data/{device_uuid}.parquet"

    def get_model_dict_path(self, device_uuid: str) -> str:
        return f"{self.output_path}/models/{device_uuid}.pkl"
