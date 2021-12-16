yolo.py : class Model(nn.Module);class Detect(nn.Module);def parse_model(d, ch); main() for test locally
common.py : 网络组件(nn.Module)，和网络推理class Detections（展示、保存等）, Classification(二级分类器)
experimental.py：实验性的网络组件(nn.Module)，集成模型class Ensemble(nn.ModuleList)，和模型权重加载方法attempt_load

tf.py: tensorflow的模型及相关方法实现