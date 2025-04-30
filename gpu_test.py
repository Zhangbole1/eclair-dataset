import torch
from MinkowskiEngine import SparseTensor, TensorField, MinkowskiAlgorithm, SparseTensorQuantizationMode

def main():
    # 检查 CUDA 可用性
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device index:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    
    # 测试 SparseTensor 是否在 GPU
    num_features = 14
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_feat = torch.randn(5, num_features).to(device)
    test_coords = torch.randint(0, 10, (5, 4)).to(device)  # [batch_idx, x, y, z]
    
    st = SparseTensor(features=test_feat, coordinates=test_coords)
    print("SparseTensor features device:", st.F.device)
    
    # 测试 TensorField -> sparse 并显式搬到 GPU
    tf = TensorField(
        features=test_feat,
        coordinates=test_coords,
        quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
    )
    sinput = tf.sparse().to(device)
    print("TensorField.sparse() features device:", sinput.F.device)

if __name__ == "__main__":
    main()
