import numpy as np
import tritonclient.http as httpclient


class TritonClient:
    def __init__(self, url: str = "localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)

    def infer(self, model_name: str, inputs: dict, outputs: list):
        infer_inputs = []

        for name, array in inputs.items():
            infer_input = httpclient.InferInput(
                name=name,
                shape=array.shape,
                datatype=self._dtype(array.dtype),
            )
            infer_input.set_data_from_numpy(array)
            infer_inputs.append(infer_input)

        infer_outputs = [httpclient.InferRequestedOutput(o) for o in outputs]

        response = self.client.infer(
            model_name=model_name,
            inputs=infer_inputs,
            outputs=infer_outputs,
        )

        return {o: response.as_numpy(o) for o in outputs}

    def get_model_metadata(self, model_name: str):
        return self.client.get_model_metadata(model_name)

    def _dtype(self, dtype):
        mapping = {
            np.int64: "INT64",
            np.int32: "INT32",
            np.float32: "FP32",
            np.bool_: "BOOL",
        }
        return mapping[dtype.type]
