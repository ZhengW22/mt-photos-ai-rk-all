from rknnlite.api import RKNNLite

# 仅这些平台支持 core_mask 参数
_CORE_MASK_SUPPORTED = {'rk3588', 'rk3576'}


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None, core_mask=RKNNLite.NPU_CORE_AUTO) -> None:
        rknn = RKNNLite()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target in _CORE_MASK_SUPPORTED:
            ret = rknn.init_runtime(core_mask=core_mask)
        else:
            ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        self.rknn = rknn

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)

        return result

    def release(self):
        self.rknn.release()
        self.rknn = None
